struct Point {
  u: f32,
  z: f32,
  f: f32,
  kappa: f32,
  grad_u: vec2<f32>,
  div_term: f32,
  pad: f32,
}

@group(0)
@binding(0)
var tex: texture_2d<f32>;

@group(0)
@binding(1)
var tex_sampler: sampler;

@group(0)
@binding(2)
var<storage, read> fluid: array<Point>;

struct VertexOutput {
    @location(0) tex_coords: vec2<f32>,
    @builtin(position) out_pos: vec4<f32>,
};


@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VertexOutput {
    let tex_coords: vec2<f32> = vec2<f32>(pos.x * 0.5 + 0.5, 1.0 - (pos.y * 0.5 + 0.5));
    let out_pos: vec4<f32> = vec4<f32>(pos, 0.0, 1.0);
    return VertexOutput(tex_coords, out_pos);
}


fn get_index(coords: vec2<u32>) -> u32 {
    let dims = vec2(512u, 512u);
    return dims.x * coords.y + coords.x;
}

fn map_color(u: f32) -> f32 {
    if u > 0.2 {
        return 4.*u;
    } else {
        return 0.;
    }
    // return exp(4. * u) - 2.;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // return textureSample(tex, tex_sampler, in.tex_coords);
    let ii: vec2<u32> = vec2<u32>(in.out_pos.xy);
    let u = fluid[get_index(ii)].u;
    let z = fluid[get_index(ii)].z;
    let col = map_color(u);
    // let col = u;
    let t = z;
    // let t_col = vec4(0., t, 0., 1.);
    let t_col = vec4(pow(t, 2.6), pow(t, 1.5), 0., 1.);
    let b_col = vec4(0., 0., 1., 1.);
    return mix(t_col, b_col, col);
}