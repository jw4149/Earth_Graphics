// This example is heavily based on the tutorial at https://open.gl

// OpenGL Helpers to reduce the clutter
#include "Helpers.h"

#include "stb_image.h"

// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>

// Linear Algebra Library
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

// Timer
#include <chrono>

using namespace Eigen;

// VertexBufferObject wrapper
VertexBufferObject VBO;
VertexBufferObject VBO_C;
VertexBufferObject VBO_N;
VertexBufferObject VBO_T;
ElementBufferObject EBO;

// Contains the vertex positions
Eigen::MatrixXf V(3,3);
Eigen::MatrixXf C(3,3);
Eigen::MatrixXf N(3,3);
Eigen::MatrixXf T(2,3);
Eigen::MatrixXi E(1,6);

Eigen::Matrix4f transformation;

MatrixXf view(4,4);
MatrixXf projection(4,4);

Vector3f eye;
Vector3f center;
Vector3f up;

float rho;
float theta;
float phi;

Vector3f light;
float angle;

int earthMode;

void lookAt(Vector3f& e, Vector3f& g, Vector3f& t, MatrixXf& view) {
    Vector3f w = -g.normalized();
    Vector3f u = (t.cross(w)).normalized();
    Vector3f v = w.cross(u);

    MatrixXf M(4,4);
    M.col(0) << u,0;
    M.col(1) << v,0;
    M.col(2) << w,0;
    M.col(3) << e,1;

    view = M.inverse();
}

void perspective(MatrixXf& projection, float aspect_ratio) {
    projection <<
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0;

    projection(0,0) = 1.0/(aspect_ratio * tan(0.3927));
    projection(1,1) = 1.0/tan(0.3927);
    projection(2,2) = - (100.0 + 0.1)/(100.0 - 0.1);
    projection(3,2) = -1;
    projection(2,3) = -(2 * 100.0 * 0.1) / (100.0 - 0.1);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    switch(key)
    {
        case GLFW_KEY_I:
        if (action == GLFW_PRESS) {
            rho = eye.norm();
            theta = acos(eye(1)/rho);
            phi = atan(eye(0)/eye(2));

            theta -= 0.05;
            eye(0) = rho * sin(theta) * sin(phi);
            eye(1) = rho * cos(theta);
            eye(2) = rho * sin(theta) * cos(phi);

            Vector3f origin(0,0,0);
            center = origin - eye;
            lookAt(eye, center, up, view);
        }
        break;

        case GLFW_KEY_K:
        if (action == GLFW_PRESS) {
            rho = eye.norm();
            theta = acos(eye(1)/rho);
            phi = atan(eye(0)/eye(2));

            theta += 0.05;
            eye(0) = rho * sin(theta) * sin(phi);
            eye(1) = rho * cos(theta);
            eye(2) = rho * sin(theta) * cos(phi);

            Vector3f origin(0,0,0);
            center = origin - eye;
            lookAt(eye, center, up, view);
        }
        break;

        case GLFW_KEY_J:
        if (action == GLFW_PRESS) {
            rho = eye.norm();
            theta = acos(eye(1)/rho);
            phi = atan(eye(0)/eye(2));

            phi -= 0.05;
            eye(0) = rho * sin(theta) * sin(phi);
            eye(1) = rho * cos(theta);
            eye(2) = rho * sin(theta) * cos(phi);

            Vector3f origin(0,0,0);
            center = origin - eye;
            lookAt(eye, center, up, view);
        }
        break;

        case GLFW_KEY_L:
        if (action == GLFW_PRESS) {
            rho = eye.norm();
            theta = acos(eye(1)/rho);
            phi = atan(eye(0)/eye(2));

            phi += 0.05;
            eye(0) = rho * sin(theta) * sin(phi);
            eye(1) = rho * cos(theta);
            eye(2) = rho * sin(theta) * cos(phi);

            Vector3f origin(0,0,0);
            center = origin - eye;
            lookAt(eye, center, up, view);
        }
        break;

        case GLFW_KEY_1:
        if (action == GLFW_PRESS) {
            earthMode = 1;
        }
        break;

        case GLFW_KEY_2:
        if (action == GLFW_PRESS) {
            earthMode = 2;
        }
        break;
    }
}

int main(void)
{
    GLFWwindow* window;

    // Initialize the library
    if (!glfwInit())
        return -1;

    // Activate supersampling
    glfwWindowHint(GLFW_SAMPLES, 8);

    // Ensure that we get at least a 3.2 context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

    // On apple we have to load a core profile with forward compatibility
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    #ifndef __APPLE__
      glewExperimental = true;
      GLenum err = glewInit();
      if(GLEW_OK != err)
      {
        /* Problem: glewInit failed, something is seriously wrong. */
       fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      }
      glGetError(); // pull and savely ignonre unhandled errors like GL_INVALID_ENUM
      fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
    #endif

    int major, minor, rev;
    major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
    printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
    printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
    printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    // Initialize the VAO
    // A Vertex Array Object (or VAO) is an object that describes how the vertex
    // attributes are stored in a Vertex Buffer Object (or VBO). This means that
    // the VAO is not the actual object storing the vertex data,
    // but the descriptor of the vertex data.
    VertexArrayObject VAO;
    VAO.init();
    VAO.bind();

    // Initialize the VBO with the vertices data
    // A VBO is a data container that lives in the GPU memory
    VBO.init();
    VBO_C.init();
    VBO_N.init();
    VBO_T.init();
    EBO.init();
/*
    V.resize(2,4);
    //V << 0,  0.5, -0.5, 0.5, -0.5, -0.5;
    V.col(0) << -0.5, 0.5;
    V.col(1) << 0.5, 0.5;
    V.col(2) << 0.5, -0.5;
    V.col(3) << -0.5, -0.5;
    VBO.update(V);

    E.resize(1,6);
    E << 0, 1, 2, 2, 3, 0;
    EBO.update(E);
*/
    int slices = 64;
    int stacks = 64;
    float pi = 3.1415926535897932384626433832795f;

    V.resize(3, (stacks+1) * (slices+1));
    C.resize(3, (stacks+1) * (slices+1));
    N.resize(3, (stacks+1) * (slices+1));
    T.resize(2, (stacks+1) * (slices+1));

    for (int i = 0; i <= stacks; ++i) {
        float v = i / (float)stacks;
        float phi = v * pi;
        for (int j = 0; j <= slices; ++j) {
            float u = j / (float)slices;
            float theta = u * 2.0f * pi;
            float x = cos(theta) * sin(phi);
            float y = cos(phi);
            float z = sin(theta) * sin(phi);
            V.col((1+slices) * i + j) << x,y,z;
            C.col((1+slices) * i + j) << 1,1,1;
            N.col((1+slices) * i + j) << x,y,z;
            T.col((1+slices) * i + j) << 1-u,v;
        }
    }

    VBO.update(V);
    VBO_C.update(C);
    VBO_N.update(N);
    VBO_T.update(T);

    E.resize(1,6*(slices * stacks + slices));
    for (int i = 0; i < slices * stacks + slices; ++i) {
        E.col(6*i) << i;
        E.col(6*i+1) << (i + slices + 1);
        E.col(6*i+2) << (i + slices);

        E.col(6*i+3) << (i + slices + 1);
        E.col(6*i+4) << i;
        E.col(6*i+5) << (i+1);
    }

    EBO.update(E);

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);   
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    /*
    int width1, height1, nrChannels1;
    unsigned char *data1 = stbi_load("earth1.jpg", &width1, &height1, &nrChannels1, 0);
    */
    int width, height, nrChannels;
    unsigned char *data = stbi_load("earth.jpg", &width, &height, &nrChannels, 0);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    
    //stbi_image_free(data1);
    stbi_image_free(data);

    // Initialize the OpenGL Program
    // A program controls the OpenGL pipeline and it must contains
    // at least a vertex shader and a fragment shader to be valid
    Program program;
    const GLchar* vertex_shader =
            "#version 150 core\n"
                    "in vec3 position;"
                    "in vec3 normal;"
                    "in vec3 color;"
                    "in vec2 texCoord;"
                    "uniform vec3 light;"
                    "uniform mat4 view;"
                    "uniform mat4 projection;"
                    "uniform mat4 transformation;"
                    "out vec3 f_color;"
                    "out vec3 f_normal;"
                    "out vec3 f_position;"
                    "out vec2 f_texCoord;"
                    "out vec3 f_light;"
                    "void main()"
                    "{"
                    "    vec4 transformed_position = transformation * vec4(position, 1.0);"
                    "    vec4 transformed_normal = normalize(transpose(inverse(transformation)) * vec4(normal, 0.0));"
                    "    gl_Position = projection * view * transformed_position;"
                    "    f_color = color;"
                    "    f_position = (view * transformed_position).xyz;"
                    "    f_normal = normalize((transpose(inverse(view)) * transformed_normal).xyz);"
                    "    f_texCoord = texCoord;"
                    "    f_light = (view * vec4(light, 1.0)).xyz;"
                    "}";
    const GLchar* fragment_shader =
            "#version 150 core\n"
                    "in vec3 f_color;"
                    "in vec3 f_normal;"
                    "in vec3 f_position;"
                    "in vec2 f_texCoord;"
                    "in vec3 f_light;"
                    "out vec4 outColor;"
                    "uniform sampler2D ourTexture;"
                    "void main()"
                    "{"
                    "    vec3 I = normalize(-f_position + f_light);"
                    "    outColor = texture(ourTexture, f_texCoord) * dot(I, f_normal);"
                    "}";

    // Compile the two shaders and upload the binary to the GPU
    // Note that we have to explicitly specify that the output "slot" called outColor
    // is the one that we want in the fragment buffer (and thus on screen)
    program.init(vertex_shader,fragment_shader,"outColor");
    program.bind();

    // The vertex shader wants the position of the vertices as an input.
    // The following line connects the VBO we defined above with the position "slot"
    // in the vertex shader
    program.bindVertexAttribArray("position",VBO);
    program.bindVertexAttribArray("color",VBO_C);
    program.bindVertexAttribArray("normal",VBO_N);
    program.bindVertexAttribArray("texCoord",VBO_T);

    earthMode = 2;

    eye << 0,0,4;
    center << 0,0,-1;
    up << 0,1,0;
    lookAt(eye, center, up, view);

    /*
    view <<
    1,0,0,0,
    0,1,0,0,
    0,0,1,0,
    0,0,0,1;
    */

    transformation <<
    1,0,0,0,
    0,1,0,0,
    0,0,1,0,
    0,0,0,1;

    light <<
    5,0,0;

    angle = 0.0;

    // Save the current time --- it will be used to dynamically change the triangle color
    auto t_start = std::chrono::high_resolution_clock::now();

    // Register the keyboard callback
    glfwSetKeyCallback(window, key_callback);

    // Register the mouse callback
    //glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
        /*
        if (earthMode == 1) {
            //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width1, height1, 0, GL_RGB, GL_UNSIGNED_BYTE, data1);
            //glGenerateMipmap(GL_TEXTURE_2D);
        } else if (earthMode == 2) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width2, height2, 0, GL_RGB, GL_UNSIGNED_BYTE, data2);
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        */

        int width, height;
        glfwGetWindowSize(window, &width, &height);
        float aspect_ratio = float(width)/float(height);

        perspective(projection, aspect_ratio);

        /*
        projection <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1;
        */

        glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, view.data());
        glUniformMatrix4fv(program.uniform("projection"), 1, GL_FALSE, projection.data());
        glUniformMatrix4fv(program.uniform("transformation"), 1, GL_FALSE, transformation.data());
        glUniform3fv(program.uniform("light"), 1, light.data());

        glUniform1i(program.uniform("ourTexture"), 0);

        // Set the uniform value depending on the time difference
        auto t_now = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();
        //Matrix4f rotate;
        transformation <<
        cos(-0.1*time),0,-sin(-0.1*time),0,
        0,1,0,0,
        sin(-0.1*time),0,cos(-0.1*time),0,
        0,0,0,1;

        light << 5.0 * cos(0.2*time), 0, 5.0 * sin(0.2*time);

        //transformation = rotate * transformation;
        //glUniform3f(program.uniform("triangleColor"), (float)(sin(time * 4.0f) + 1.0f) / 2.0f, 0.0f, 0.0f);
        // Clear the framebuffer
        //glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        //VAO.bind();
        glBindTexture(GL_TEXTURE_2D, texture);
        // Draw a triangle
        glDrawElements(GL_TRIANGLES, ( slices * stacks + slices ) * 6, GL_UNSIGNED_INT, 0);

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Deallocate opengl memory
    program.free();
    VAO.free();
    VBO.free();
    EBO.free();

    // Deallocate glfw internals
    glfwTerminate();
    return 0;
}
