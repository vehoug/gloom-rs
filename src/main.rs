// Uncomment these following global attributes to silence most warnings of "low" interest:
/*
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_unsafe)]
#![allow(unused_variables)]
*/
extern crate nalgebra_glm as glm;
use std::{ mem, ptr, os::raw::c_void };
use std::thread;
use std::sync::{Mutex, Arc, RwLock};

mod shader;
mod util;
mod mesh;

use glutin::event::{Event, WindowEvent, DeviceEvent, KeyboardInput, ElementState::{Pressed, Released}, VirtualKeyCode::{self, *}};
use glutin::event_loop::ControlFlow;

// initial window size
const INITIAL_SCREEN_W: u32 = 800;
const INITIAL_SCREEN_H: u32 = 600;

// == // Helper functions to make interacting with OpenGL a little bit prettier. You *WILL* need these! // == //

// Get the size of an arbitrary array of numbers measured in bytes
// Example usage:  byte_size_of_array(my_array)
fn byte_size_of_array<T>(val: &[T]) -> isize {
    std::mem::size_of_val(&val[..]) as isize
}

// Get the OpenGL-compatible pointer to an arbitrary array of numbers
// Example usage:  pointer_to_array(my_array)
fn pointer_to_array<T>(val: &[T]) -> *const c_void {
    &val[0] as *const T as *const c_void
}

// Get the size of the given type in bytes
// Example usage:  size_of::<u64>()
fn size_of<T>() -> i32 {
    mem::size_of::<T>() as i32
}

// Get an offset in bytes for n units of type T, represented as a relative pointer
// Example usage:  offset::<u64>(4)
fn offset<T>(n: u32) -> *const c_void {
    (n * mem::size_of::<T>() as u32) as *const T as *const c_void
}

// Get a null pointer (equivalent to an offset of 0)
// ptr::null()


// Generate a Vertex Array Object (VAO) and return its ID
unsafe fn create_vao(vertices: &Vec<f32>, indices: &Vec<u32>, colors: &Vec<f32>, normals: &Vec<f32>) -> u32 {
    let pos_entry_size: i32 = 3;
    let color_entry_size: i32 = 4;
    let normal_entry_size: i32 = 3;

    // Create and bind the VAO
    let mut array_id: u32 = 0;
    gl::GenVertexArrays(1, &mut array_id);
    gl::BindVertexArray(array_id);

    // Create and bind position VBO and fill with vertex position data
    let mut vertex_buffer_id: u32 = 0;
    gl::GenBuffers(1, &mut vertex_buffer_id);
    gl::BindBuffer(gl::ARRAY_BUFFER, vertex_buffer_id);
    gl::BufferData(gl::ARRAY_BUFFER, 
                   byte_size_of_array(vertices), 
                   pointer_to_array(vertices), 
                   gl::STATIC_DRAW);
    
    // Set the Vertex Attribute Pointer for positions and enable it
    let vert_attrib_index: u32 = 0;
    gl::VertexAttribPointer(vert_attrib_index, 
                            pos_entry_size, 
                            gl::FLOAT, 
                            gl::FALSE, 
                            pos_entry_size * size_of::<f32>(), 
                            std::ptr::null());
    gl::EnableVertexAttribArray(vert_attrib_index);

    // Create and bind color VBO and fill with vertex color data
    let mut color_buffer_id: u32 = 0;
    gl::GenBuffers(1, &mut color_buffer_id);
    gl::BindBuffer(gl::ARRAY_BUFFER, color_buffer_id);
    gl::BufferData(gl::ARRAY_BUFFER,
                   byte_size_of_array(colors),
                   pointer_to_array(colors),
                   gl::STATIC_DRAW);

    // Set the Vertex Attribute Pointer for colors and enable it
    let color_attrib_index: u32 = 1;
    gl::VertexAttribPointer(color_attrib_index, 
                            color_entry_size, 
                            gl::FLOAT, 
                            gl::FALSE, 
                            color_entry_size * size_of::<f32>(), 
                            std::ptr::null());
    gl::EnableVertexAttribArray(color_attrib_index);

    // Create and bind normal VBO and fill with vertex normal data
    let mut normal_buffer_id: u32 = 0;
    gl::GenBuffers(1, &mut normal_buffer_id);
    gl::BindBuffer(gl::ARRAY_BUFFER, normal_buffer_id);
    gl::BufferData(gl::ARRAY_BUFFER,
                   byte_size_of_array(normals),
                   pointer_to_array(normals),
                   gl::STATIC_DRAW);

    // Set the Vertex Attribute Pointer for normals and enable it
    let normal_attrib_index: u32 = 2;
    gl::VertexAttribPointer(normal_attrib_index, 
                            normal_entry_size, 
                            gl::FLOAT, 
                            gl::FALSE, 
                            normal_entry_size * size_of::<f32>(), 
                            std::ptr::null());
    gl::EnableVertexAttribArray(normal_attrib_index);

    // Create and bind the Index Buffer before filling it with index data
    let mut index_buffer_id: u32 = 0;
    gl::GenBuffers(1, &mut index_buffer_id);
    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer_id);
    gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, 
                   byte_size_of_array(indices), 
                   pointer_to_array(indices), 
                   gl::STATIC_DRAW);

    // Return the VAO ID
    array_id
}


fn main() {
    // Set up the necessary objects to deal with windows and event handling
    let el = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Gloom-rs")
        .with_resizable(true)
        .with_inner_size(glutin::dpi::LogicalSize::new(INITIAL_SCREEN_W, INITIAL_SCREEN_H));
    let cb = glutin::ContextBuilder::new()
        .with_vsync(true);
    let windowed_context = cb.build_windowed(wb, &el).unwrap();
    // Uncomment these if you want to use the mouse for controls, but want it to be confined to the screen and/or invisible.
    // windowed_context.window().set_cursor_grab(true).expect("failed to grab cursor");
    // windowed_context.window().set_cursor_visible(false);

    // Set up a shared vector for keeping track of currently pressed keys
    let arc_pressed_keys = Arc::new(Mutex::new(Vec::<VirtualKeyCode>::with_capacity(10)));
    // Make a reference of this vector to send to the render thread
    let pressed_keys = Arc::clone(&arc_pressed_keys);

    // Set up shared tuple for tracking mouse movement between frames
    let arc_mouse_delta = Arc::new(Mutex::new((0f32, 0f32)));
    // Make a reference of this tuple to send to the render thread
    let mouse_delta = Arc::clone(&arc_mouse_delta);

    // Set up shared tuple for tracking changes to the window size
    let arc_window_size = Arc::new(Mutex::new((INITIAL_SCREEN_W, INITIAL_SCREEN_H, false)));
    // Make a reference of this tuple to send to the render thread
    let window_size = Arc::clone(&arc_window_size);

    // Spawn a separate thread for rendering, so event handling doesn't block rendering
    let render_thread = thread::spawn(move || {
        // Acquire the OpenGL Context and load the function pointers.
        // This has to be done inside of the rendering thread, because
        // an active OpenGL context cannot safely traverse a thread boundary
        let context = unsafe {
            let c = windowed_context.make_current().unwrap();
            gl::load_with(|symbol| c.get_proc_address(symbol) as *const _);
            c
        };

        let mut window_aspect_ratio = INITIAL_SCREEN_W as f32 / INITIAL_SCREEN_H as f32;

        // Set up openGL
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);
            gl::Enable(gl::CULL_FACE);
            gl::Disable(gl::MULTISAMPLE);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::DebugMessageCallback(Some(util::debug_callback), ptr::null());

            // Print some diagnostics
            println!("{}: {}", util::get_gl_string(gl::VENDOR), util::get_gl_string(gl::RENDERER));
            println!("OpenGL\t: {}", util::get_gl_string(gl::VERSION));
            println!("GLSL\t: {}", util::get_gl_string(gl::SHADING_LANGUAGE_VERSION));
        }

        // == // Set up your VAO around here

        // Define vertex data for triangles
        let vertices: &Vec<f32> = &vec![
            -0.7, -0.1, 0.0,  -0.5, -0.1, 0.0,  -0.6, 0.1, 0.0,
             0.5, -0.1, 0.0,   0.7, -0.1, 0.0,   0.6, 0.1, 0.0,
            -0.1,  0.4, 0.0,   0.1,  0.4, 0.0,   0.0, 0.6, 0.0,
            -0.1, -0.6, 0.0,   0.1, -0.6, 0.0,   0.0, -0.4, 0.0
        ];

        let colors: &Vec<f32> = &vec![
            1.0, 0.0, 0.0, 1.0,   0.0, 1.0, 0.0, 1.0,   0.0, 0.0, 1.0, 1.0,
            1.0, 1.0, 0.0, 0.9,   0.0, 1.0, 1.0, 0.9,   1.0, 0.0, 1.0, 0.9,
            1.0, 0.5, 0.5, 0.8,   0.5, 1.0, 0.5, 0.8,   0.5, 0.5, 1.0, 0.8,
            1.0, 1.0, 1.0, 0.7,   0.5, 0.5, 0.5, 0.7,   0.2, 0.2, 0.2, 0.7
        ];

        // Define index data for triangles
        let indices: &Vec<u32> = &vec![
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11
        ];

        // Create single VAO containing all triangles
        // let vao = unsafe { create_vao(vertices, indices, colors) };

        // Define relative paths to the simple shader files
        let vertex_shader_path: &str = "./shaders/simple.vert";
        let fragment_shader_path: &str = "./shaders/simple.frag";

        // Define relative paths to the model files
        let lunar_surface_path: &str = "./resources/lunarsurface.obj";
        let helicopter_path: &str = "./resources/helicopter.obj";

        // Load the lunar surface model mesh 
        let lunar_surface = mesh::Terrain::load(lunar_surface_path);

        let vao = unsafe { create_vao(&lunar_surface.vertices, 
                                           &lunar_surface.indices, 
                                           &lunar_surface.colors,
                                           &lunar_surface.normals) 
        };

        // Create the simple shader object
        let simple_shader = unsafe {
            shader::ShaderBuilder::new()
                .attach_file(vertex_shader_path)
                .attach_file(fragment_shader_path)
                .link()
        };

        // Initialize camera pose: position (x, y, z) and rotation (pitch, yaw) w/o roll
        let mut camera_pose: Vec<f32> = vec![
            0.0, 0.0, 0.0,  // Position: x, y, z
            0.0, 0.0        // Rotation: pitch, yaw
        ];

        // The main rendering loop
        let first_frame_time = std::time::Instant::now();
        let mut previous_frame_time = first_frame_time;
        loop {
            // Compute time passed since the previous frame and since the start of the program
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(first_frame_time).as_secs_f32();
            let delta_time = now.duration_since(previous_frame_time).as_secs_f32();
            previous_frame_time = now;

            // Handle resize events
            if let Ok(mut new_size) = window_size.lock() {
                if new_size.2 {
                    context.resize(glutin::dpi::PhysicalSize::new(new_size.0, new_size.1));
                    window_aspect_ratio = new_size.0 as f32 / new_size.1 as f32;
                    (*new_size).2 = false;
                    println!("Window was resized to {}x{}", new_size.0, new_size.1);
                    unsafe { gl::Viewport(0, 0, new_size.0 as i32, new_size.1 as i32); }
                }
            }

            // Handle keyboard input
            if let Ok(keys) = pressed_keys.lock() {
                for key in keys.iter() {
                    match key {
                        // The `VirtualKeyCode` enum is defined here:
                        //    https://docs.rs/winit/0.25.0/winit/event/enum.VirtualKeyCode.html

                        VirtualKeyCode::A => {
                            camera_pose[0] -= 1.0 * delta_time;
                        }
                        VirtualKeyCode::D => {
                            camera_pose[0] += 1.0 * delta_time;
                        }
                        VirtualKeyCode::W => {
                            camera_pose[2] -= 1.0 * delta_time;
                        }
                        VirtualKeyCode::S => {
                            camera_pose[2] += 1.0 * delta_time;
                        }
                        VirtualKeyCode::Space => {
                            camera_pose[1] += 1.0 * delta_time;
                        }
                        VirtualKeyCode::LShift => {
                            camera_pose[1] -= 1.0 * delta_time;
                        }
                        VirtualKeyCode::Right => {
                            camera_pose[3] += 1.0 * delta_time;
                        }
                        VirtualKeyCode::Left => {
                            camera_pose[3] -= 1.0 * delta_time;
                        }
                        VirtualKeyCode::Down => {
                            camera_pose[4] += 1.0 * delta_time;
                        }
                        VirtualKeyCode::Up => {
                            camera_pose[4] -= 1.0 * delta_time;
                        }

                        // default handler:
                        _ => { }
                    }
                }
            }
            // Handle mouse movement. delta contains the x and y movement of the mouse since last frame in pixels
            if let Ok(mut delta) = mouse_delta.lock() {

                // == // Optionally access the accumulated mouse movement between
                // == // frames here with `delta.0` and `delta.1`

                *delta = (0.0, 0.0); // reset when done
            }

            // == // Please compute camera transforms here (exercise 2 & 3)
            
            // FOV in the y-direction
            let fov_y: f32 = 75.0;
            
            // Initialize transformation matrix as identity each frame
            let mut transform: glm::Mat4 = glm::identity();

            // Translate the scene in the -z direction to move it into the view frustum
            let translate_z: glm::Mat4 = glm::translation(&glm::vec3(0.0, 0.0, -2.0));

            // Compute camera translation and rotation from keyboard inputs
            let camera_translate: glm::Mat4 = glm::translation(&glm::vec3(-camera_pose[0], 
                                                                          -camera_pose[1],  
                                                                          -camera_pose[2]));

            let camera_rotate: glm::Mat4 = glm::rotation(-camera_pose[3], &glm::vec3(0.0, 1.0, 0.0))
                                         * glm::rotation(-camera_pose[4], &glm::vec3(1.0, 0.0, 0.0));

            // Compute the perspective projection matrix
            let projection: glm::Mat4 = glm::perspective(window_aspect_ratio, 
                                                           fov_y.to_radians(), 
                                                           1.0, 1000.0);

            // Compute final transformation with matrix multiplication
            transform = projection * camera_rotate * camera_translate * translate_z * transform;

            unsafe {
                // Clear the color and depth buffers
                gl::ClearColor(0.035, 0.046, 0.078, 1.0); // night sky
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

                // Activate the shader program
                simple_shader.activate();

                // == // Issue the necessary gl:: commands to draw your scene here

                // Upload transformation matrix to the vertex shader
                let location = simple_shader.get_uniform_location("transform");
                gl::UniformMatrix4fv(location, 1, gl::FALSE, transform.as_ptr());
                
                gl::BindVertexArray(vao);
                
                // Draw all triangles from VAO
                gl::DrawElements(gl::TRIANGLES, lunar_surface.indices.len() as i32, gl::UNSIGNED_INT, std::ptr::null());
                

            }

            // Display the new color buffer on the display
            context.swap_buffers().unwrap(); // we use "double buffering" to avoid artifacts
        }
    });


    // == //
    // == // From here on down there are only internals.
    // == //


    // Keep track of the health of the rendering thread
    let render_thread_healthy = Arc::new(RwLock::new(true));
    let render_thread_watchdog = Arc::clone(&render_thread_healthy);
    thread::spawn(move || {
        if !render_thread.join().is_ok() {
            if let Ok(mut health) = render_thread_watchdog.write() {
                println!("Render thread panicked!");
                *health = false;
            }
        }
    });

    // Start the event loop -- This is where window events are initially handled
    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        // Terminate program if render thread panics
        if let Ok(health) = render_thread_healthy.read() {
            if *health == false {
                *control_flow = ControlFlow::Exit;
            }
        }

        match event {
            Event::WindowEvent { event: WindowEvent::Resized(physical_size), .. } => {
                println!("New window size received: {}x{}", physical_size.width, physical_size.height);
                if let Ok(mut new_size) = arc_window_size.lock() {
                    *new_size = (physical_size.width, physical_size.height, true);
                }
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            // Keep track of currently pressed keys to send to the rendering thread
            Event::WindowEvent { event: WindowEvent::KeyboardInput {
                    input: KeyboardInput { state: key_state, virtual_keycode: Some(keycode), .. }, .. }, .. } => {

                if let Ok(mut keys) = arc_pressed_keys.lock() {
                    match key_state {
                        Released => {
                            if keys.contains(&keycode) {
                                let i = keys.iter().position(|&k| k == keycode).unwrap();
                                keys.remove(i);
                            }
                        },
                        Pressed => {
                            if !keys.contains(&keycode) {
                                keys.push(keycode);
                            }
                        }
                    }
                }

                // Handle Escape and Q keys separately
                match keycode {
                    Escape => { *control_flow = ControlFlow::Exit; }
                    Q      => { *control_flow = ControlFlow::Exit; }
                    _      => { }
                }
            }
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                // Accumulate mouse movement
                if let Ok(mut position) = arc_mouse_delta.lock() {
                    *position = (position.0 + delta.0 as f32, position.1 + delta.1 as f32);
                }
            }
            _ => { }
        }
    });
}
