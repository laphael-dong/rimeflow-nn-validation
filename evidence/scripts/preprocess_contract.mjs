import { createHash } from 'node:crypto';

export const PREPROCESS_CONTRACT = Object.freeze({
  kind: 'letterbox',
  targetSize: [640, 640],
  interpolation: 'linear',
  samplerAddressMode: 'clamp-to-edge',
  outputPixelCoordinate: 'u=x/targetWidth; v=y/targetHeight',
  sourceTexelCoordinate: 'texel=normalizedCoordinate*sourceSize-0.5',
  color: {
    sourceChannels: ['R', 'G', 'B', 'A'],
    modelChannels: ['R', 'G', 'B'],
    alpha: 'ignored',
  },
  valueRange: { minimum: 0, maximum: 1, inclusive: true },
  normalize: { kind: 'none', mean: null, standardDeviation: null, scale: null, offset: null },
  letterbox: {
    scaleRule: 'min(targetWidth/sourceWidth,targetHeight/sourceHeight)',
    scaledDimensionRounding: 'none-continuous',
    paddingRule: 'symmetric-continuous',
    padXRule: '(1-sourceWidth*scale/targetWidth)/2',
    padYRule: '(1-sourceHeight*scale/targetHeight)/2',
    insideRule: '0<=u<=1 && 0<=v<=1',
    fillRgb: [0.447, 0.447, 0.447],
  },
});

export function readPpm(bytes) {
  const marker = Buffer.from('\n255\n');
  const headerEnd = bytes.indexOf(marker);
  if (headerEnd < 0) throw new Error('unsupported PPM header');
  const header = bytes.subarray(0, headerEnd).toString('ascii').trim().split(/\s+/);
  if (header[0] !== 'P6') throw new Error('only P6 PPM is supported');
  const width = Number(header[1]);
  const height = Number(header[2]);
  const pixels = bytes.subarray(headerEnd + marker.length);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) throw new Error('invalid PPM dimensions');
  if (pixels.length !== width * height * 3) throw new Error('PPM byte length mismatch');
  return { width, height, pixels };
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function sampleLinear(image, normalizedX, normalizedY, channel) {
  const texelX = normalizedX * image.width - 0.5;
  const texelY = normalizedY * image.height - 0.5;
  const x0 = Math.floor(texelX);
  const y0 = Math.floor(texelY);
  const fx = texelX - x0;
  const fy = texelY - y0;
  const value = (x, y) => image.pixels[(clamp(y, 0, image.height - 1) * image.width + clamp(x, 0, image.width - 1)) * 3 + channel] / 255;
  const top = value(x0, y0) * (1 - fx) + value(x0 + 1, y0) * fx;
  const bottom = value(x0, y0 + 1) * (1 - fx) + value(x0 + 1, y0 + 1) * fx;
  return top * (1 - fy) + bottom * fy;
}

export function preprocessCanonical(image) {
  const [targetWidth, targetHeight] = PREPROCESS_CONTRACT.targetSize;
  const scale = Math.min(targetWidth / image.width, targetHeight / image.height);
  const padXNormalized = (1 - image.width * scale / targetWidth) / 2;
  const padYNormalized = (1 - image.height * scale / targetHeight) / 2;
  const regionWidth = 1 - 2 * padXNormalized;
  const regionHeight = 1 - 2 * padYNormalized;
  const tensor = new Float32Array(3 * targetWidth * targetHeight);
  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const outputX = x / targetWidth;
      const outputY = y / targetHeight;
      const sourceX = (outputX - padXNormalized) / regionWidth;
      const sourceY = (outputY - padYNormalized) / regionHeight;
      const targetOffset = y * targetWidth + x;
      const inside = sourceX >= 0 && sourceX <= 1 && sourceY >= 0 && sourceY <= 1;
      for (let channel = 0; channel < 3; channel++) {
        tensor[channel * targetWidth * targetHeight + targetOffset] = inside
          ? sampleLinear(image, sourceX, sourceY, channel)
          : PREPROCESS_CONTRACT.letterbox.fillRgb[channel];
      }
    }
  }
  return {
    tensor,
    scale,
    padXNormalized,
    padYNormalized,
    padXPixels: padXNormalized * targetWidth,
    padYPixels: padYNormalized * targetHeight,
  };
}

export function tensorDigest(tensor) {
  return createHash('sha256').update(Buffer.from(tensor.buffer, tensor.byteOffset, tensor.byteLength)).digest('hex');
}
