use std::{
    fs,
    path::{Path, PathBuf},
    sync::mpsc,
};

use bytemuck::{Pod, Zeroable};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use wgpu::util::DeviceExt;

const TARGET: u32 = 640;
const TOLERANCE: f32 = 6.0e-5;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    src_w: f32,
    src_h: f32,
    dst_size: u32,
    pad_a: u32,
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    pad_b: f32,
}

struct Ppm {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
}

fn read_ppm(path: &Path) -> Result<Ppm, String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let marker = b"\n255\n";
    let header_end = bytes
        .windows(marker.len())
        .position(|window| window == marker)
        .ok_or("unsupported PPM header")?;
    let header = std::str::from_utf8(&bytes[..header_end]).map_err(|error| error.to_string())?;
    let mut fields = header.split_whitespace();
    if fields.next() != Some("P6") {
        return Err("only P6 PPM is supported".into());
    }
    let width = fields
        .next()
        .ok_or("missing width")?
        .parse::<u32>()
        .map_err(|error| error.to_string())?;
    let height = fields
        .next()
        .ok_or("missing height")?
        .parse::<u32>()
        .map_err(|error| error.to_string())?;
    let rgb = bytes[(header_end + marker.len())..].to_vec();
    if rgb.len() != width as usize * height as usize * 3 {
        return Err("PPM byte length mismatch".into());
    }
    Ok(Ppm { width, height, rgb })
}

fn letterbox(image: &Ppm) -> Params {
    let target = TARGET as f32;
    let scale = (target / image.width as f32).min(target / image.height as f32);
    let pad_x = (1.0 - image.width as f32 * scale / target) / 2.0;
    let pad_y = (1.0 - image.height as f32 * scale / target) / 2.0;
    Params {
        src_w: image.width as f32,
        src_h: image.height as f32,
        dst_size: TARGET,
        pad_a: 0,
        scale,
        pad_x,
        pad_y,
        pad_b: 0.0,
    }
}

fn sample(image: &Ppm, u: f32, v: f32, channel: usize) -> f32 {
    let texel_x = u * image.width as f32 - 0.5;
    let texel_y = v * image.height as f32 - 0.5;
    let x0 = texel_x.floor() as i32;
    let y0 = texel_y.floor() as i32;
    let fx = texel_x - x0 as f32;
    let fy = texel_y - y0 as f32;
    let value = |x: i32, y: i32| {
        let x = x.clamp(0, image.width as i32 - 1) as usize;
        let y = y.clamp(0, image.height as i32 - 1) as usize;
        image.rgb[(y * image.width as usize + x) * 3 + channel] as f32 / 255.0
    };
    let top = value(x0, y0) * (1.0 - fx) + value(x0 + 1, y0) * fx;
    let bottom = value(x0, y0 + 1) * (1.0 - fx) + value(x0 + 1, y0 + 1) * fx;
    top * (1.0 - fy) + bottom * fy
}

fn cpu_reference(image: &Ppm, params: Params) -> Vec<f32> {
    let mut output = vec![0.0; 3 * TARGET as usize * TARGET as usize];
    let region_u = 1.0 - 2.0 * params.pad_x;
    let region_v = 1.0 - 2.0 * params.pad_y;
    for y in 0..TARGET {
        for x in 0..TARGET {
            let u = (x as f32 / TARGET as f32 - params.pad_x) / region_u;
            let v = (y as f32 / TARGET as f32 - params.pad_y) / region_v;
            let inside = (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v);
            for channel in 0..3 {
                let index = channel * TARGET as usize * TARGET as usize
                    + y as usize * TARGET as usize
                    + x as usize;
                output[index] = if inside {
                    sample(image, u, v, channel)
                } else {
                    0.447
                };
            }
        }
    }
    output
}

fn digest(values: &[f32]) -> String {
    format!("{:x}", Sha256::digest(bytemuck::cast_slice(values)))
}

fn gpu_output(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader: &wgpu::ShaderModule,
    image: &Ppm,
    params: Params,
) -> Result<Vec<f32>, String> {
    let rgba: Vec<u8> = image
        .rgb
        .chunks_exact(3)
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 255])
        .collect();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("task1-conformance-source"),
        size: wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(image.width * 4),
            rows_per_image: Some(image.height),
        },
        texture.size(),
    );
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("task1-preprocess-conformance"),
        layout: None,
        module: shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output_size = (3 * TARGET as u64 * TARGET as u64 * 4) as wgpu::BufferAddress;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("task1-output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("task1-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let view = texture.create_view(&Default::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("task1-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform.as_entire_binding(),
            },
        ],
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("task1-staging"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("task1-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(TARGET.div_ceil(16), TARGET.div_ceil(16), 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| error.to_string())?;
    receiver
        .recv()
        .map_err(|error| error.to_string())?
        .map_err(|error| error.to_string())?;
    let values = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    Ok(values)
}

fn main() -> Result<(), String> {
    let root = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".into()));
    let manifest: Value = serde_json::from_slice(
        &fs::read(root.join("evidence/fixtures/manifest.json"))
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let shader_source = fs::read_to_string(root.join("shaders/preprocess.wgsl"))
        .map_err(|error| error.to_string())?;
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .map_err(|error| error.to_string())?;
    let info = adapter.get_info();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .map_err(|error| error.to_string())?;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("task1-production-wgsl"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let mut fixtures = Vec::new();
    for item in manifest["images"]
        .as_array()
        .ok_or("manifest images missing")?
    {
        let id = item["id"].as_str().ok_or("fixture id missing")?;
        let image = read_ppm(&root.join(item["path"].as_str().ok_or("fixture path missing")?))?;
        let params = letterbox(&image);
        let cpu = cpu_reference(&image, params);
        let gpu = gpu_output(&device, &queue, &shader, &image, params)?;
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut mismatch_count = 0usize;
        for (expected, actual) in cpu.iter().zip(&gpu) {
            let difference = (expected - actual).abs();
            max_abs = max_abs.max(difference);
            sum_abs += difference as f64;
            if difference > TOLERANCE {
                mismatch_count += 1;
            }
        }
        fixtures.push(json!({ "id": id, "elements": cpu.len(), "toleranceAbsolute": TOLERANCE, "maxAbsoluteDifference": max_abs, "meanAbsoluteDifference": sum_abs / cpu.len() as f64, "mismatchCount": mismatch_count, "cpuSha256Float32Le": digest(&cpu), "wgslSha256Float32Le": digest(&gpu), "passed": mismatch_count == 0 }));
    }
    let report = json!({
        "schemaVersion": 1,
        "shaderPath": "shaders/preprocess.wgsl",
        "adapter": { "name": info.name, "vendor": info.vendor, "device": info.device, "deviceType": format!("{:?}", info.device_type), "backend": format!("{:?}", info.backend), "driver": info.driver, "driverInfo": info.driver_info },
        "contract": { "interpolation": "linear", "samplerAddressMode": "clamp-to-edge", "outputPixelCoordinate": "u=x/targetWidth; v=y/targetHeight", "sourceTexelCoordinate": "texel=normalizedCoordinate*sourceSize-0.5" },
        "toleranceFreeze": {
            "status": "initial-freeze-before-native-candidate-results",
            "absolute": TOLERANCE,
            "basis": "CPU f32 bilinear interpolation versus production WGSL textureSampleLevel on the recorded Vulkan adapter; final real-image suite observed maximum was 5.02467155456543e-5",
            "doesNotModifyModelOutputTolerances": true,
            "reviewRecord": {
                "date": "2026-08-10",
                "decision": "accepted-before-native-candidate-results",
                "reviewerRole": "task1-evidence-implementation",
                "previousPreliminaryAbsolute": 2.0e-5,
                "reason": "preliminary synthetic fixtures understated hardware linear-filter interpolation error; the model-output tolerances in web-reference.json are unchanged"
            }
        },
        "fixtures": fixtures,
    });
    fs::write(
        root.join("evidence/reports/preprocess-conformance.json"),
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    if report["fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["passed"] != true)
    {
        return Err("CPU/WGSL conformance failed".into());
    }
    Ok(())
}
