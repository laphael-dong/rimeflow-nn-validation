use wasm_bindgen::prelude::*;

#[wasm_bindgen(inline_js = r#"
// ─── GPUDevice 拦截 ───
// Intercept ALL requestDevice calls so wgpu and ORT share the SAME device.
// wgpu creates the device first; when ORT later calls requestDevice, it
// gets the same device back instead of creating a new one.
let _capturedDevice = null;
let _origRequestDevice = null;

export function capture_webgpu_device() {
    if (_origRequestDevice || typeof GPUAdapter === 'undefined') return;
    _origRequestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function(...args) {
        if (_capturedDevice) {
            console.log('[ort-bridge] returning shared WebGPU device to caller');
            return _capturedDevice;
        }
        const device = await _origRequestDevice.apply(this, args);
        _capturedDevice = device;
        console.log('[ort-bridge] captured WebGPU device (first caller)');
        return device;
    };
}

// ─── GPU 预处理管线 ───
let _preprocessPipeline = null;
let _preprocessBuffer = null;
let _paramsBuffer = null;
const DST_SIZE = 640;

const PREPROCESS_WGSL = `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
struct Params { src_w: f32, src_h: f32, dst_size: u32, _pad: u32, scale: f32, pad_x: f32, pad_y: f32, _pad2: f32 }
@group(0) @binding(3) var<uniform> p: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let x = gid.x; let y = gid.y;
    if (x >= p.dst_size || y >= p.dst_size) { return; }
    let out_u = f32(x) / f32(p.dst_size);
    let out_v = f32(y) / f32(p.dst_size);
    let in_u = (out_u - p.pad_x) / p.scale;
    let in_v = (out_v - p.pad_y) / p.scale;
    var pixel: vec4f;
    if (in_u < 0.0 || in_u > 1.0 || in_v < 0.0 || in_v > 1.0) {
        pixel = vec4f(0.447, 0.447, 0.447, 1.0);
    } else {
        pixel = textureSampleLevel(src, src_sampler, vec2f(in_u, in_v), 0.0);
    }
    let hw = p.dst_size * p.dst_size;
    let idx = y * p.dst_size + x;
    dst[0u * hw + idx] = pixel.r;
    dst[1u * hw + idx] = pixel.g;
    dst[2u * hw + idx] = pixel.b;
}
`;

function _initPreprocessPipeline(device) {
    const module = device.createShaderModule({ code: PREPROCESS_WGSL });
    _preprocessPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
    });
    const bufSize = 3 * DST_SIZE * DST_SIZE * 4;
    _preprocessBuffer = device.createBuffer({
        size: bufSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    _paramsBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
}

// ─── ORT Session ───
let _session = null;
let _inputName = null;

export async function ort_init(model_url) {
    if (typeof globalThis.ort === 'undefined') {
        console.log('[ort-bridge] loading onnxruntime-web from /ort/ort.min.js');
        await new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src = '/ort/ort.min.js';
            s.onload = resolve;
            s.onerror = () => reject(new Error('Failed to load /ort/ort.min.js'));
            document.head.appendChild(s);
        });
        if (typeof globalThis.ort === 'undefined') {
            throw new Error('onnxruntime-web failed to initialize globalThis.ort');
        }
    }
    globalThis.ort.env.wasm.wasmPaths = '/ort/';
    globalThis.ort.env.wasm.numThreads = 1;
    // GPU zero-copy preprocessing disabled: fromGpuBuffer with shared device
    // silently fails on ORT 1.27 (no error but empty output). CPU preprocessing
    // + ORT WebGPU EP inference still gives GPU-accelerated results.
    // TODO: investigate ORT fromGpuBuffer compatibility.
    const useGpuPreprocess = false;
    const resp = await fetch(model_url);
    if (!resp.ok) throw new Error('Model fetch failed: ' + resp.status + ' ' + model_url);
    const buffer = await resp.arrayBuffer();
    _session = await globalThis.ort.InferenceSession.create(buffer, {
        executionProviders: ['webgpu', 'wasm'],
    });
    _inputName = _session.inputNames[0];
    if (useGpuPreprocess && _capturedDevice && !_preprocessPipeline) {
        _initPreprocessPipeline(_capturedDevice);
    }
    console.log('[ort-bridge] session ready, input=' + _inputName + ', gpuPreprocess=' + useGpuPreprocess);
}

// ─── 推理 ───
export async function ort_detect(canvas) {
    if (!_session) throw new Error('ORT session not initialized');
    const useGpu = !!(_capturedDevice && _preprocessPipeline);

    if (useGpu && _capturedDevice && _preprocessPipeline) {
        // === GPU zero-copy 路径 ===
        const device = _capturedDevice;
        const srcW = canvas.width, srcH = canvas.height;
        const scale = Math.min(DST_SIZE / srcW, DST_SIZE / srcH);
        const padX = (1 - (srcW * scale) / DST_SIZE) / 2;
        const padY = (1 - (srcH * scale) / DST_SIZE) / 2;

        const tex = device.createTexture({
            size: [srcW, srcH], format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
                 | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        device.queue.copyExternalImageToTexture(
            { source: canvas }, { texture: tex }, [srcW, srcH],
        );
        device.queue.writeBuffer(_paramsBuffer, 0, new Float32Array([
            srcW, srcH, DST_SIZE, 0, scale, padX, padY, 0,
        ]));
        const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
        const bg = device.createBindGroup({
            layout: _preprocessPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: tex.createView() },
                { binding: 1, resource: sampler },
                { binding: 2, resource: { buffer: _preprocessBuffer } },
                { binding: 3, resource: { buffer: _paramsBuffer } },
            ],
        });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(_preprocessPipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(DST_SIZE / 16), Math.ceil(DST_SIZE / 16));
        pass.end();
        device.queue.submit([enc.finish()]);
        tex.destroy();

        const tensor = globalThis.ort.Tensor.fromGpuBuffer(_preprocessBuffer, {
            dataType: 'float32', dims: [1, 3, DST_SIZE, DST_SIZE],
        });
        const result = await _session.run({ [_inputName]: tensor });
        const output = await result[_session.outputNames[0]].getData();
        return { output, scale, padX: padX * DST_SIZE, padY: padY * DST_SIZE, srcW, srcH };
    }

    // === CPU fallback 路径 (WebGL) ===
    const offscreen = new OffscreenCanvas(DST_SIZE, DST_SIZE);
    const ctx = offscreen.getContext('2d');
    const srcW = canvas.width, srcH = canvas.height;
    const scale = Math.min(DST_SIZE / srcW, DST_SIZE / srcH);
    const dw = srcW * scale, dh = srcH * scale;
    const dx = (DST_SIZE - dw) / 2, dy = (DST_SIZE - dh) / 2;
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, DST_SIZE, DST_SIZE);
    ctx.drawImage(canvas, 0, 0, srcW, srcH, dx, dy, dw, dh);
    const imageData = ctx.getImageData(0, 0, DST_SIZE, DST_SIZE);
    const data = new Float32Array(3 * DST_SIZE * DST_SIZE);
    const hw = DST_SIZE * DST_SIZE;
    for (let i = 0; i < hw; i++) {
        data[i]          = imageData.data[i * 4]     / 255;
        data[hw + i]     = imageData.data[i * 4 + 1] / 255;
        data[2 * hw + i] = imageData.data[i * 4 + 2] / 255;
    }
    const tensor = new globalThis.ort.Tensor('float32', data, [1, 3, DST_SIZE, DST_SIZE]);
    const result = await _session.run({ [_inputName]: tensor });
    const output = await result[_session.outputNames[0]].getData();
    return { output, scale, padX: dx, padY: dy, srcW, srcH };
}

export function ort_release() {
    if (_session) { _session.release(); _session = null; _inputName = null; }
    if (_preprocessBuffer) { _preprocessBuffer.destroy(); _preprocessBuffer = null; }
    if (_paramsBuffer) { _paramsBuffer.destroy(); _paramsBuffer = null; }
    _preprocessPipeline = null;
}
"#)]
extern "C" {
    pub fn capture_webgpu_device();

    #[wasm_bindgen(catch)]
    pub async fn ort_init(model_url: &str) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(catch)]
    pub async fn ort_detect(canvas: &web_sys::HtmlCanvasElement) -> Result<JsValue, JsValue>;

    pub fn ort_release();
}

pub fn get_output_f32(result: &JsValue, key: &str) -> Option<Vec<f32>> {
    let val = js_sys::Reflect::get(result, &JsValue::from_str(key)).ok()?;
    Some(js_sys::Float32Array::from(val).to_vec())
}

pub fn get_f64(result: &JsValue, key: &str) -> f64 {
    js_sys::Reflect::get(result, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}
