// use nannou::prelude::*;

use bytemuck;
use nannou::image;
use nannou::image::GenericImageView;
use nannou::prelude::*;
use nannou::{color::Rgb, image::Pixel};

type Dims = [u32; 2];
const N: u32 = 512;
const DIMS: Dims = [N, N];
const SIZE: u32 = DIMS[0] * DIMS[1];

fn color_map(x: f32) -> Rgb {
    assert!(x.is_finite());
    Rgb::new(0., 1. - x, x)
}

fn get_index(i: u32, j: u32) -> usize {
    (j * DIMS[0] + i) as usize
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 2],
}

const VERTICES: [Vertex; 4] = [
    Vertex {
        position: [-1.0, 1.0],
    },
    Vertex {
        position: [-1.0, -1.0],
    },
    Vertex {
        position: [1.0, 1.0],
    },
    Vertex {
        position: [1.0, -1.0],
    },
];

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
struct Point {
    u: f32,
    z: f32,
    f: f32,
    kappa: f32,
    grad_u: [f32; 2],
    div_term: f32,
    pad: f32,
}

use nannou::wgpu::BufferInitDescriptor;
use std::{
    sync::{Arc, Mutex},
    thread::sleep,
    time::Duration,
};

struct Model {
    compute: Compute,
    render: Render,
    fluid: Arc<Mutex<Vec<Point>>>,
    iter: usize,
}

struct Compute {
    fluid_buffer: wgpu::Buffer,
    fluid_size: wgpu::BufferAddress,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

struct Render {
    vertex_buffer: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

const SCREEN_SIZE: u32 = DIMS[0];

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    dst_format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    wgpu::RenderPipelineBuilder::from_layout(layout, shader)
        .fragment_shader(shader)
        .vertex_entry_point("vs_main")
        .fragment_entry_point("fs_main")
        .color_format(dst_format)
        .add_vertex_buffer::<Vertex>(&wgpu::vertex_attr_array![0 => Float32x2])
        .sample_count(sample_count)
        .primitive_topology(wgpu::PrimitiveTopology::TriangleStrip)
        .build(device)
}

fn main() {
    nannou::app(model)
        .update(update)
        .capture_frame_timeout(Some(Duration::from_secs_f32(60.)))
        .run();
}

fn create_bind_group_layout(
    device: &wgpu::Device,
    texture_sample_type: wgpu::TextureSampleType,
    sampler_filtering: bool,
) -> wgpu::BindGroupLayout {
    wgpu::BindGroupLayoutBuilder::new()
        .texture(
            wgpu::ShaderStages::FRAGMENT,
            false,
            wgpu::TextureViewDimension::D2,
            texture_sample_type,
        )
        .sampler(wgpu::ShaderStages::FRAGMENT, sampler_filtering)
        .storage_buffer(
            wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            false,
            true,
        )
        .build(device)
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    texture: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    fluid: &wgpu::Buffer,
    fluid_size: wgpu::BufferAddress,
) -> wgpu::BindGroup {
    let buffer_size_bytes = std::num::NonZeroU64::new(fluid_size).unwrap();
    wgpu::BindGroupBuilder::new()
        .texture_view(texture)
        .sampler(sampler)
        .buffer_bytes(fluid, 0, Some(buffer_size_bytes))
        .build(device, layout)
}

fn model(app: &App) -> Model {
    let image = image::open("snowdon3.png").unwrap();
    let (img_w, img_h) = image.dimensions();

    app.set_loop_mode(LoopMode::NTimes {
        number_of_updates: 4000,
    });
    let w_id = app
        .new_window()
        .size_pixels(SCREEN_SIZE, SCREEN_SIZE)
        .view(view)
        .build()
        .unwrap();
    let window = app.window(w_id).unwrap();
    let device = window.device();

    // Create the compute shader module.
    let cs_mod = nannou_wgpu::shader_from_spirv_bytes(&device, include_bytes!("shader.spv"));
    let render_mod = nannou_wgpu::shader_from_spirv_bytes(&device, include_bytes!("render.spv"));

    let mut fluid = vec![
        Point {
            u: 0.01,
            z: 0.,
            f: 0.,
            grad_u: [0., 2.],
            kappa: 0.,
            div_term: 0.,
            pad: 0.,
        };
        SIZE as usize
    ];
    let ig = image.as_rgb8().unwrap();
    for i in 0..DIMS[0] {
        for j in 0..DIMS[1] {
            let f = &mut fluid[get_index(i, j)];
            f.z = (ig.get_pixel(i, j).to_rgb().0[0] as f32) / (u8::MAX as f32);
            let x = i as f32 / (DIMS[0] as f32);
            let y = j as f32 / (DIMS[1] as f32);
            let xx = x - 0.5_f32;
            let yy = y - 0.5_f32;
            let a = (0.2) * (-xx * xx * 30.).exp() * (-yy * yy * 30.).exp() + 0.01;
            f.f = a;
        }
    }
    // for i in 0..DIMS[0] {
    //     for j in 0..DIMS[1] {
    //         let f = &mut fluid[get_index(i, j)];
    //         let x = i as f32 / (DIMS[0] as f32);
    //         let y = j as f32 / (DIMS[1] as f32);
    //         let xx = x - 0.5_f32;
    //         let yy = y - 0.5_f32;
    //         let a = (10.) * (-xx * xx * 50.).exp() * (-yy * yy * 50.).exp() + 0.01;
    //         f.f = a * 10.;
    //         let ix = (i as f32) / (DIMS[0] as f32);
    //         let jx = (j as f32) / (DIMS[0] as f32);
    //         f.z = ((jx * PI * 5.).sin() + 1.) * ((ix * 10.).cos() + 1.);
    //     }
    // }

    // Create the buffer that will store the result of our compute operation.
    let fluid_size = (SIZE as usize * std::mem::size_of::<Point>()) as wgpu::BufferAddress;
    let casted = bytemuck::cast_slice(&fluid[..]);
    let fluid_buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
        label: Some("fluid"),
        contents: casted,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // Create the buffer that will store time.
    let uniforms = DIMS;
    let uniforms_bytes = uniforms_as_bytes(&uniforms);
    let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("uniform-buffer"),
        contents: uniforms_bytes,
        usage,
    });

    // Create the bind group and pipeline.
    let bind_group_layout_compute = create_bind_group_layout_compute(device);
    let bind_group_compute = create_bind_group_compute(
        device,
        &bind_group_layout_compute,
        &fluid_buffer,
        fluid_size,
        &uniform_buffer,
    );
    let pipeline_layout_compute = create_pipeline_layout(device, &bind_group_layout_compute);
    let pipeline_compute = create_compute_pipeline(device, &pipeline_layout_compute, &cs_mod);

    let format = Frame::TEXTURE_FORMAT;
    let msaa_samples = window.msaa_samples();
    let texture = wgpu::Texture::from_image(&window, &image);
    let texture_view = texture.view().build();

    let sampler_desc = wgpu::SamplerBuilder::new().into_descriptor();
    let sampler_filtering = wgpu::sampler_filtering(&sampler_desc);
    let sampler = device.create_sampler(&sampler_desc);

    let bind_group_layout =
        create_bind_group_layout(device, texture_view.sample_type(), sampler_filtering);
    let bind_group = create_bind_group(
        device,
        &bind_group_layout,
        &texture_view,
        &sampler,
        &fluid_buffer,
        fluid_size,
    );
    let pipeline_layout = create_pipeline_layout(device, &bind_group_layout);
    let render_pipeline =
        create_render_pipeline(device, &pipeline_layout, &render_mod, format, msaa_samples);

    let vertices_bytes = vertices_as_bytes(&VERTICES[..]);
    let usage = wgpu::BufferUsages::VERTEX;
    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: vertices_bytes,
        usage,
    });

    let compute = Compute {
        fluid_buffer,
        fluid_size,
        uniform_buffer,
        bind_group: bind_group_compute,
        pipeline: pipeline_compute,
    };

    let render = Render {
        bind_group,
        vertex_buffer,
        render_pipeline,
    };

    // The vector that we will write oscillator values to.

    let fluidr = Arc::new(Mutex::new(fluid));
    // let fluid_ref = fluidr.clone();
    Model {
        compute,
        render,
        fluid: fluidr.clone(),
        iter: 0,
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    model.iter += 1;
    let window = app.main_window();
    let device = window.device();
    let win_rect = window.rect();
    let compute = &mut model.compute;

    // The buffer into which we'll read some data.
    // let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    //     label: Some("read-fluid"),
    //     size: compute.fluid_size,
    //     usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    //     mapped_at_creation: false,
    // });

    // An update for the uniform buffer with the current time.
    // let uniforms = DIMS;
    // let uniforms_size = std::mem::size_of::<Dims>() as wgpu::BufferAddress;
    // let uniforms_bytes = uniforms_as_bytes(&uniforms);
    // let usage = wgpu::BufferUsages::COPY_SRC;
    // let new_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
    //     label: Some("uniform-data-transfer"),
    //     contents: uniforms_bytes,
    //     usage,
    // });

    // The encoder we'll use to encode the compute pass.
    let desc = wgpu::CommandEncoderDescriptor {
        label: Some("fluid-compute"),
    };
    let mut encoder = device.create_command_encoder(&desc);
    // encoder.copy_buffer_to_buffer(
    //     &new_uniform_buffer,
    //     0,
    //     &compute.uniform_buffer,
    //     0,
    //     uniforms_size,
    // );
    {
        let pass_desc = wgpu::ComputePassDescriptor {
            label: Some("nannou-wgpu_compute_shader-compute_pass"),
        };

        let mut cpass = encoder.begin_compute_pass(&pass_desc);
        cpass.set_pipeline(&compute.pipeline);
        cpass.set_bind_group(0, &compute.bind_group, &[]);
        cpass.dispatch(DIMS[0] as u32 / 8, DIMS[1] as u32 / 8, 1);
    }
    // encoder.copy_buffer_to_buffer(
    //     &compute.fluid_buffer,
    //     0,
    //     &read_buffer,
    //     0,
    //     compute.fluid_size,
    // );

    // // Submit the compute pass to the device's queue.
    window.queue().submit(Some(encoder.finish()));

    // Spawn a future that reads the result of the compute pass.
    // let fluid_ref = model.fluid.clone();
    // let future = async move {
    //     let slice = read_buffer.slice(..);
    //     if let Ok(_) = slice.map_async(wgpu::MapMode::Read).await {
    //         if let Ok(mut fluid) = fluid_ref.lock() {
    //             let bytes = &slice.get_mapped_range()[..];
    //             // "Cast" the slice of bytes to a slice of floats as required.
    //             let floats = {
    //                 let len = bytes.len() / std::mem::size_of::<Point>();
    //                 let ptr = bytes.as_ptr() as *const Point;
    //                 unsafe { std::slice::from_raw_parts(ptr, len) }
    //             };
    //             fluid.copy_from_slice(floats);
    //         }
    //     }
    // };
    // async_std::task::spawn(future);

    // pollster::block_on(future);

    // Check for resource cleanups and mapping callbacks.
    //
    // Note that this line is not necessary in our case, as the device we are using already gets
    // polled when nannou submits the command buffer for drawing and presentation after `view`
    // completes. If we were to use a standalone device to create our buffer and perform our
    // compute (rather than the device requested during window creation), calling `poll` regularly
    // would be a must.
    //
    // device.poll(false);
}

fn view(app: &App, model: &Model, frame: Frame) {
    // sleep(Duration::from_secs_f32(1.));
    // if model.iter > 10 {
    //     return;
    // }
    {
        let mut encoder = frame.command_encoder();
        let mut render_pass = wgpu::RenderPassBuilder::new()
            .color_attachment(frame.texture_view(), |color| color)
            .begin(&mut encoder);
        render_pass.set_bind_group(0, &model.render.bind_group, &[]);
        render_pass.set_pipeline(&model.render.render_pipeline);
        render_pass.set_vertex_buffer(0, model.render.vertex_buffer.slice(..));
        let vertex_range = 0..VERTICES.len() as u32;
        let instance_range = 0..1;
        render_pass.draw(vertex_range, instance_range);
    }

    // let draw = app.draw();
    // let window = app.window(frame.window_id()).unwrap();
    // let rect = window.rect();
    // let rr = Vec2::new(rect.w(), rect.h());
    // let cursor_p = app.mouse.position();
    // let cursor_pp = app.mouse.position() + rr / 2.;
    // let d = Vec2::new(DIMS[0] as f32, DIMS[1] as f32);

    // let r = (cursor_pp / rr) * d;
    // let ii = r.x as u32;
    // let jj = r.y as u32;

    // let idxx = get_index(ii, jj);
    // if let Ok(fluid) = model.fluid.lock() {
    //     let uu = fluid[idxx].u;
    //     draw.text(&format!("{:.2}", uu))
    //         .xy(cursor_p + Vec2::ONE * 20.)
    //         .font_size(30);
    // }
    // else {
    //     panic!("CANT lOCK");
    // }

    // if let Ok(fluid) = model.fluid.lock() {
    //     // for (i, &osc) in fluid.iter().enumerate() {

    //     // }
    //     let w = (rect.w() as f32) / (DIMS[0] as f32);
    //     let h = (rect.h() as f32) / (DIMS[1] as f32);

    //     // let (min_p, max_p) = (-500e4, 10e4);

    //     for i in 0..DIMS[0] {
    //         for j in 0..DIMS[1] {
    //             let p = fluid[get_index(i as u32, j as u32)];
    //             let x = w * (i as f32 - (DIMS[0] / 2) as f32 + 0.5);
    //             let y = h * (j as f32 - (DIMS[1] / 2) as f32 + 0.5);
    //             let xy = Vec2::new(x, y);
    //             // assert!(p.kappa.is_finite());
    //             // if !p.u.is_finite() {
    //             //     dbg!(p);
    //             // }
    //             // assert!(p.u.is_finite());

    //             let uv = Vec2::new(p.grad_u[0], p.grad_u[1]);

    //             // assert!(uv.x.is_finite());
    //             // assert!(uv.y.is_finite());

    //             let grad_u = Vec2::new(p.grad_u[0], p.grad_u[1]);
    //             // assert!(grad_u.y.is_finite());
    //             // assert!(grad_u.x.is_finite());

    //             // let color = color_map((p.u + 1.2) / 4.);
    //             let color = color_map(p.z);
    //             draw.rect().xy(xy).w_h(w, h).color(color).z(-0.01);

    //             let l = if uv.length() > 0.00001 {
    //                 (10. * uv).clamp_length(1., 10.)
    //             } else {
    //                 Vec2::ZERO
    //             };
    //             let n = 4;
    //             if (i % n == 0) && (j % n == 0) {
    //                 //     draw.arrow()
    //                 //     .start(xy)
    //                 //     .end(xy + l)
    //                 //     .color(LIGHTGREEN)
    //                 //     .weight(1.5);
    //             }
    //         }
    //     }
    // }

    // draw.to_frame(app, &frame).unwrap();
    // if model.iter > 150 {
    // app.main_window()
    //     .capture_frame(format!("flood_frames/{:04}.png", model.iter));
    // }
    dbg!(model.iter);
    sleep(Duration::from_secs_f32(0.04));
}

// fn create_uniforms(dims: Dims) -> Uniforms {
//     Uniforms {
//         dims,
//     }
// }

fn create_bind_group_layout_compute(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage_dynamic = false;
    let storage_readonly = false;
    let uniform_dynamic = false;
    wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(
            wgpu::ShaderStages::COMPUTE,
            storage_dynamic,
            storage_readonly,
        )
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, uniform_dynamic)
        .build(device)
}

fn create_bind_group_compute(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    oscillator_buffer: &wgpu::Buffer,
    oscillator_buffer_size: wgpu::BufferAddress,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let buffer_size_bytes = std::num::NonZeroU64::new(oscillator_buffer_size).unwrap();
    wgpu::BindGroupBuilder::new()
        .buffer_bytes(oscillator_buffer, 0, Some(buffer_size_bytes))
        .buffer::<Dims>(uniform_buffer, 0..1)
        .build(device, layout)
}

fn create_pipeline_layout(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("nannou"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    cs_mod: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let desc = wgpu::ComputePipelineDescriptor {
        label: Some("nannou"),
        layout: Some(layout),
        module: &cs_mod,
        entry_point: "main",
    };
    device.create_compute_pipeline(&desc)
}

// See `nannou::wgpu::bytes` docs for why these are necessary.

fn uniforms_as_bytes(uniforms: &Dims) -> &[u8] {
    unsafe { wgpu::bytes::from(uniforms) }
}

fn vertices_as_bytes(data: &[Vertex]) -> &[u8] {
    unsafe { wgpu::bytes::from_slice(data) }
}
