#version 430 core

in layout(location = 0) vec3 position;
in layout(location = 1) vec4 color;

out vec4 vert_color;

uniform layout(location = 2) float a;

// Note: Column-major order
mat4x4 transform = {{1, a, 0, 0}, 
                    {0, 1, 0, 0}, 
                    {0, 0, 1, 0}, 
                    {0, 0, 0, 1}};

void main()
{
    gl_Position = transform * vec4(position, 1.0f);
    vert_color = color;
}