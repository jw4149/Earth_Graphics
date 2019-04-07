#include "Helpers.h"

#include "stb_image.h"

#include <GLFW/glfw3.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>


#include <chrono>

using namespace Eigen;

VertexBufferObject VBO;
VertexBufferObject VBO_C;
VertexBufferObject VBO_N;
VertexBufferObject VBO_T;
ElementBufferObject EBO;

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
    }
}

int main(void)
{
    GLFWwindow* window;

    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_SAMPLES, 8);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    #ifndef __APPLE__
      glewExperimental = true;
      GLenum err = glewInit();
      if(GLEW_OK != err)
      {
       fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      }
      glGetError(); 
      fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
    #endif

    int major, minor, rev;
    major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
    printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
    printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
    printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    VertexArrayObject VAO;
    VAO.init();
    VAO.bind();

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

    // sphere constructs
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

    //loading textures

    unsigned int texture1;
    glGenTextures(1, &texture1);
    glBindTexture(GL_TEXTURE_2D, texture1);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);   
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    /*
    int width1, height1, nrChannels1;
    unsigned char *data1 = stbi_load("earth1.jpg", &width1, &height1, &nrChannels1, 0);
    */
    int width, height, nrChannels;
    unsigned char *data = stbi_load("../pics/earth.jpg", &width, &height, &nrChannels, 0);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    
    //stbi_image_free(data1);
    stbi_image_free(data);

    unsigned int texture2;
    glGenTextures(1, &texture2);
    glBindTexture(GL_TEXTURE_2D, texture2);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);   
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    /*
    int width1, height1, nrChannels1;
    unsigned char *data1 = stbi_load("earth1.jpg", &width1, &height1, &nrChannels1, 0);
    */
    //int width, height, nrChannels;
    data = stbi_load("../pics/earth_night.jpg", &width, &height, &nrChannels, 0);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    
    //stbi_image_free(data1);
    stbi_image_free(data);

    //pipeline
    Program program;
    const GLchar* vertex_shader =
            "#version 150 core\n"
                    "in vec3 position;"
                    "in vec3 normal;"
                    "in vec3 color;"
                    "in vec2 texCoord;"
                    "uniform vec3 light;"
                    "uniform vec3 camera;"
                    "uniform mat4 view;"
                    "uniform mat4 projection;"
                    "uniform mat4 transformation;"
                    "out vec3 f_color;"
                    "out vec3 f_normal;"
                    "out vec3 f_position;"
                    "out vec2 f_texCoord;"
                    "out vec3 f_light;"
                    "out vec3 f_camera;"
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
                    "    f_camera = (view * vec4(camera, 1.0)).xyz;"
                    "}";
    const GLchar* fragment_shader =
            "#version 150 core\n"
                    "in vec3 f_color;"
                    "in vec3 f_normal;"
                    "in vec3 f_position;"
                    "in vec2 f_texCoord;"
                    "in vec3 f_light;"
                    "in vec3 f_camera;"
                    "out vec4 outColor;"
                    "uniform sampler2D texture1;"
                    "uniform sampler2D texture2;"
                    "void main()"
                    "{"
                    "    vec3 I = normalize(-f_position + f_light);"
                    "    vec3 v = normalize(-f_position + f_camera);"
                    "    vec3 h = (I+v)/2;"
                    "    outColor = mix(texture(texture2, f_texCoord), texture(texture1, f_texCoord), 1.5*clamp(dot(I, f_normal), 0.15, 1.0));"
                    "}";

    program.init(vertex_shader,fragment_shader,"outColor");
    program.bind();

    program.bindVertexAttribArray("position",VBO);
    program.bindVertexAttribArray("color",VBO_C);
    program.bindVertexAttribArray("normal",VBO_N);
    program.bindVertexAttribArray("texCoord",VBO_T);

    eye << 0,0,2.8;
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
    100,0,0;

    angle = -1.5;

    // current time
    auto t_start = std::chrono::high_resolution_clock::now();

    glfwSetKeyCallback(window, key_callback);

    glUniform1i(program.uniform("texture1"), 0);
    glUniform1i(program.uniform("texture2"), 1);

    // Loop until window closed
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
        glUniform3fv(program.uniform("camera"), 1, eye.data());

        // ratation animation
        auto t_now = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();
        //Matrix4f rotate;
        transformation <<
        cos(-0.1*time),0,-sin(-0.1*time),0,
        0,1,0,0,
        sin(-0.1*time),0,cos(-0.1*time),0,
        0,0,0,1;

        light << 5.0 * cos(angle + 0.3*time), 0, 5.0 * sin(angle + 0.3*time);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);
        
        glDrawElements(GL_TRIANGLES, ( slices * stacks + slices ) * 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    program.free();
    VAO.free();
    VBO.free();
    EBO.free();

    glfwTerminate();
    return 0;
}
