
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/******************************************************************************************************/
#ifdef __APPLE__
#include <mach/mach_time.h>
uint64_t _start, _end;
mach_timebase_info_data_t sTimebaseInfo;
#define CHECK_TIME_START _start = mach_absolute_time()
#define CHECK_TIME_END(a) _end = mach_absolute_time();  if (sTimebaseInfo.denom == 0) { mach_timebase_info(&sTimebaseInfo); }  \
a = (_end - _start) * sTimebaseInfo.numer / sTimebaseInfo.denom * 1.0e-6f
#elseif __WIN32__
#include <Windows.h>
__int64 _start, _freq, _end;
#define CHECK_TIME_START QueryPerformanceFrequency((LARGE_INTEGER*)&_freq); QueryPerformanceCounter((LARGE_INTEGER*)&_start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((LARGE_INTEGER*)&_end); a = (float)((float)(_end - _start) / (_freq * 1.0e-3f))
#else
#include <chrono>
auto __check_time_start = std::chrono::time_point<std::chrono::steady_clock>();
auto __check_time_end = std::chrono::time_point<std::chrono::steady_clock>();
#define CHECK_TIME_START                                    \
    __check_time_start = std::chrono::steady_clock::now();
#define CHECK_TIME_END(DURATION)                                        \
    __check_time_end = std::chrono::steady_clock::now();                \
    DURATION = std::chrono::duration_cast<std::chrono::milliseconds>(   \
        __check_time_end - __check_time_start).count();
#endif

/******************************************************************************************************/
// For OpenCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#include <CL/cl_gl.h>
#endif

#include "my_OpenCL_util.h"
#include "cpu_implementation.h"

typedef struct _OPENCL_C_PROG_SRC {
    size_t length;
    char *string;
} OPENCL_C_PROG_SRC;

#define OPENCL_C_PROG_POS_FILE_NAME "programs/cloth_position.cl"
#define OPENCL_C_PROG_NOR_FILE_NAME "programs/cloth_normal.cl"
#define KERNEL_POS_NAME "cloth_position"
#define KERNEL_NOR_NAME "cloth_normal"

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue cmd_queue;
cl_program program[2];
cl_kernel kernel[2];

cl_mem buf_pos[2];
cl_mem buf_vel[2];
cl_mem buf_normal;

/******************************************************************************************************/
// For OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <FreeImage.h>

#ifdef __linux__
#include <GL/glx.h>
#endif

#include "shaders/LoadShaders.h"

const int WINDOW_WIDTH  = 800;
const int WINDOW_HEIGHT = 600;

/******************************************************************************************************/
typedef struct _Light_Parameters {
    int light_on;
    float position[4];
    float ambient_color[4], diffuse_color[4], specular_color[4];
    float spot_direction[3];
    float spot_exponent;
    float spot_cutoff_angle;
    float light_attenuation_factors[4]; // produce this effect only if .w != 0.0f
} Light_Parameters;

typedef struct _loc_LIGHT_Parameters {
    GLint light_on;
    GLint position;
    GLint ambient_color, diffuse_color, specular_color;
    GLint spot_direction;
    GLint spot_exponent;
    GLint spot_cutoff_angle;
    GLint light_attenuation_factors;
} loc_light_Parameters;

typedef struct _Material_Parameters {
    float ambient_color[4], diffuse_color[4], specular_color[4], emissive_color[4];
    float specular_exponent;
} Material_Parameters;

typedef struct _loc_Material_Parameters {
    GLint ambient_color, diffuse_color, specular_color, emissive_color;
    GLint specular_exponent;
} loc_Material_Parameters;

GLuint h_ShaderProgram_Phong; // handles to shader programs

// for Phong Shading (Textured) shaders
#define NUMBER_OF_LIGHT_SUPPORTED 1
GLint loc_global_ambient_color;
loc_light_Parameters loc_light[NUMBER_OF_LIGHT_SUPPORTED];
loc_Material_Parameters loc_material;
GLint loc_ModelViewProjectionMatrix_TXPS, loc_ModelViewMatrix_TXPS, loc_ModelViewMatrixInvTrans_TXPS;
GLint loc_texture;

GLuint loc_curr_pos, loc_next_pos;
GLuint loc_curr_vel, loc_next_vel;
GLuint loc_normal;
GLuint loc_element_indices, loc_texcoord;

GLuint cloth_VAO;
GLuint cloth_BOs[7];

// include glm/*.hpp only if necessary
//#include <glm/glm.hpp> 
#include <glm/gtc/matrix_transform.hpp> //translate, rotate, scale, lookAt, perspective, etc.
#include <glm/gtc/matrix_inverse.hpp> // inverseTranspose, etc.
glm::mat4 ModelViewProjectionMatrix, ModelViewMatrix;
glm::mat3 ModelViewMatrixInvTrans;
glm::mat4 ViewMatrix, ProjectionMatrix;

#define TO_RADIAN 0.01745329252f  
#define TO_DEGREE 57.295779513f
#define BUFFER_OFFSET(offset) ((GLvoid *) (offset))

#define LOC_VERTEX 0
#define LOC_NORMAL 1
#define LOC_TEXCOORD 2

// lights in scene
Light_Parameters light[NUMBER_OF_LIGHT_SUPPORTED];

// texture stuffs
#define N_TEXTURES_USED 1
#define TEXTURE_ID_CLOTH 0
GLuint texture_names[N_TEXTURES_USED];

/******************************************************************************************************/
// For cloth simulation

#include "parameters.h"
float REST_LENGTH_HORIZ;
float REST_LENGTH_VERT;
float REST_LENGTH_DIAG;
int NUM_ITER = 500;
float DELTA_T = (1.0f/ NUM_ITER)*(1.0f / 60.0f); // Draw cloth every once per 1/60 sec.

/******************************************************************************************************/

char* strnstr(const char *s, const char *find, size_t slen) {
    char c, sc;
    size_t len;

    if ((c = *find++) != '\0') {
        len = strlen(find);
        do {
            do {
                if (slen-- < 1 || (sc = *s++) == '\0')
                    return (NULL);
            } while (sc != c);
            if (len > slen)
                return (NULL);
        } while (strncmp(s, find, len) != 0);
        s--;
    }
    return ((char *)s);
}

bool IsExtensionSupported(const char* support_str, const char* ext_string, size_t ext_buffer_size) {
    size_t offset = 0;
    const char* space_substr = strnstr(ext_string + offset, " ", ext_buffer_size - offset);
    size_t space_pos = space_substr ? space_substr - ext_string : 0;
    while (space_pos < ext_buffer_size) {
        if (strncmp(support_str, ext_string + offset, space_pos) == 0) {
            // Device supports requested extension!
            printf("Info: Found extension support ¡®%s¡¯!\n", support_str);
            return true;
        }
        // Keep searching -- skip to next token string
        offset = space_pos + 1;
        space_substr = strnstr(ext_string + offset, " ", ext_buffer_size - offset);
        space_pos = space_substr ? space_substr - ext_string : 0;
    }
    printf("Warning: Extension not supported ¡®%s¡¯!\n", support_str);
    return false;
}

void InitializeBuffers() {
    cl_int errcode_ret;

    // Initial transform
    glm::mat4 transf = glm::translate(glm::mat4(1.0), glm::vec3(0, CLOTH_SIZE_Y, 0));
    transf = glm::rotate(transf, glm::radians(-80.0f), glm::vec3(1, 0, 0));
    transf = glm::translate(transf, glm::vec3(0, -CLOTH_SIZE_Y, 0));

    // Initial positions of the particles
    int buffer_size = NUM_PARTICLES_X * NUM_PARTICLES_Y;
    GLfloat* init_position = (GLfloat*)malloc(4 * buffer_size * sizeof(GLfloat));
    GLfloat* init_velocity = (GLfloat*)malloc(4 * buffer_size * sizeof(GLfloat));
    memset(init_velocity, 0, 4 * buffer_size * sizeof(GLfloat));

    float* init_texcoord = (float*)malloc(2 * buffer_size * sizeof(float));
    float dx = CLOTH_SIZE_X / (NUM_PARTICLES_X - 1);
    float dy = CLOTH_SIZE_Y / (NUM_PARTICLES_Y - 1);
    float ds = 1.0f / (NUM_PARTICLES_X - 1);
    float dt = 1.0f / (NUM_PARTICLES_Y - 1);

    int pos_idx = 0, tc_idx = 0;
    glm::vec4 p(0.0f, 0.0f, 0.0f, 1.0f);
    for (int i = 0; i < NUM_PARTICLES_Y; i++) {
        for (int j = 0; j < NUM_PARTICLES_X; j++) {
            p.x = dx * j;
            p.y = dy * i;
            p.z = 0.0f;
            p = transf * p;

            init_position[pos_idx++] = p.x;
            init_position[pos_idx++] = p.y;
            init_position[pos_idx++] = p.z;
            init_position[pos_idx++] = 1.0f;

            init_texcoord[tc_idx++] = ds * j;
            init_texcoord[tc_idx++] = dt * i;
        }
    }

    // Every row is one triangle strip
    int el_idx = 0;
    GLuint* element_index = (GLuint*)malloc((2 * buffer_size + NUM_PARTICLES_Y) * sizeof(GLuint));
    for (int row = 0; row < NUM_PARTICLES_Y - 1; row++) {
        for (int col = 0; col < NUM_PARTICLES_X; col++) {
            element_index[el_idx++] = ((row + 1) * NUM_PARTICLES_X + (col));
            element_index[el_idx++] = ((row)* NUM_PARTICLES_X + (col));
        }
        element_index[el_idx++] = PRIM_RESTART;
    }

    // We need buffers for position (2), element index,
    // velocity (2), normal, and texture coordinates.
    glGenBuffers(7, cloth_BOs);
    loc_curr_pos = cloth_BOs[0];
    loc_next_pos = cloth_BOs[1];
    loc_curr_vel = cloth_BOs[2];
    loc_next_vel = cloth_BOs[3];
    loc_normal = cloth_BOs[4];
    loc_element_indices = cloth_BOs[5];
    loc_texcoord = cloth_BOs[6];

    // The buffers for positions
    glBindBuffer(GL_ARRAY_BUFFER, loc_curr_pos);
    glBufferData(GL_ARRAY_BUFFER, 4 * buffer_size * sizeof(GLfloat), &init_position[0], GL_DYNAMIC_DRAW);
    buf_pos[0] = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, loc_curr_pos, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);

    glBindBuffer(GL_ARRAY_BUFFER, loc_next_pos);
    glBufferData(GL_ARRAY_BUFFER, 4 * buffer_size * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    buf_pos[1] = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, loc_next_pos, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);

    // Velocities
    glBindBuffer(GL_ARRAY_BUFFER, loc_curr_vel);
    glBufferData(GL_ARRAY_BUFFER, 4 * buffer_size * sizeof(GLfloat), &init_velocity[0], GL_DYNAMIC_COPY);
    buf_vel[0] = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, loc_curr_vel, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);

    glBindBuffer(GL_ARRAY_BUFFER, loc_next_vel);
    glBufferData(GL_ARRAY_BUFFER, 4 * buffer_size * sizeof(GLfloat), NULL, GL_DYNAMIC_COPY);
    buf_vel[1] = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, loc_next_vel, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);

    // Normal buffer
    glBindBuffer(GL_ARRAY_BUFFER, loc_normal);
    glBufferData(GL_ARRAY_BUFFER, 4 * buffer_size * sizeof(GLfloat), NULL, GL_DYNAMIC_COPY);
    buf_normal = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, loc_normal, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);

    // Element indicies
    glBindBuffer(GL_ARRAY_BUFFER, loc_element_indices);
    glBufferData(GL_ARRAY_BUFFER, (2 * buffer_size + NUM_PARTICLES_Y) * sizeof(GLuint), &element_index[0], GL_DYNAMIC_COPY);

    // Texture coordinates
    glBindBuffer(GL_ARRAY_BUFFER, loc_texcoord);
    glBufferData(GL_ARRAY_BUFFER, 2 * buffer_size * sizeof(float), &init_texcoord[0], GL_STATIC_DRAW);

    free(init_position);
    free(init_velocity);
    free(element_index);
    free(init_texcoord);

    // Set up the VAO
    glGenVertexArrays(1, &cloth_VAO);
    glBindVertexArray(cloth_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, loc_curr_pos);
    glVertexAttribPointer(LOC_VERTEX, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(LOC_VERTEX);

    glBindBuffer(GL_ARRAY_BUFFER, loc_normal);
    glVertexAttribPointer(LOC_NORMAL, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(LOC_NORMAL);

    glBindBuffer(GL_ARRAY_BUFFER, loc_texcoord);
    glVertexAttribPointer(LOC_TEXCOORD, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(LOC_TEXCOORD);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, loc_element_indices);
    glBindVertexArray(0);
}

bool InitializeOpenCL() {
    cl_int errcode_ret;
    float compute_time;

    OPENCL_C_PROG_SRC prog_src_position, prog_src_normal;

    if (0) {
        // Just to reveal my OpenCl platform...
        show_OpenCL_platform();
        return false;
    }

    /* Get the first platform. */
    errcode_ret = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR_CODE(errcode_ret);

    /* Get the first GPU device. */
    errcode_ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR_CODE(errcode_ret);

    /* Get string containing supported device extensions. */
    size_t ext_size = 1024;
    char* ext_string = (char*)malloc(ext_size);
    errcode_ret = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_size, ext_string, &ext_size);

    /* Search for GL support in extension string (space delimited). */
    int supported = IsExtensionSupported(CL_GL_SHARING_EXT, ext_string, ext_size);
    if (!supported)
    {
        // Device dosen't support context sharing with OpenGL
        printf("Not Found GL Sharing Support!\n");
        return false;
    }

    /* Create a context with the devices. */
#ifdef __APPLE__
    // Get current CGL Context and CGL Share group
    CGLContextObj kCGLContext = CGLGetCurrentContext();
    CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);

    // Create CL context properties, add handle & share-group enum
    cl_context_properties properties[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, 
        (cl_context_properties)kCGLShareGroup, 0
    };
#elseif __WIN32__
    // Create CL context properties, add WGL context & handle to DC
    cl_context_properties properties[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), // WGL Context
        CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), // WGL HDC
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, // OpenCL platform
        0
    };
#else
    cl_context_properties properties[] =
        {
            CL_GL_CONTEXT_KHR,   (cl_context_properties)glXGetCurrentContext(),
            CL_GLX_DISPLAY_KHR,  (cl_context_properties)glXGetCurrentDisplay(),
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform, // OpenCL platform object
            0
        };
#endif

    /* Create a context with the devices. */
    context = clCreateContext(properties, 1, &device, NULL, NULL, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);

    /* Create a command-queue for the GPU device. */
    cmd_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);

    /* Create a program from OpenCL C source code. */
    prog_src_position.length = read_kernel_from_file(OPENCL_C_PROG_POS_FILE_NAME, &prog_src_position.string);
    program[0] = clCreateProgramWithSource(context, 1, (const char **)&prog_src_position.string, &prog_src_position.length, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);

    prog_src_normal.length = read_kernel_from_file(OPENCL_C_PROG_NOR_FILE_NAME, &prog_src_normal.string);
    program[1] = clCreateProgramWithSource(context, 1, (const char **)&prog_src_normal.string, &prog_src_normal.length, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);

    /* Build a program executable from the program object. */
    errcode_ret = clBuildProgram(program[0], 1, &device, NULL, NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        print_build_log(program[0], device, "GPU");
        return false;
    }

    /* Build a program executable from the program object. */
    errcode_ret = clBuildProgram(program[1], 1, &device, NULL, NULL, NULL);
    if (errcode_ret != CL_SUCCESS) {
        print_build_log(program[1], device, "GPU");
        return false;
    }

    /* Create the kernel from the program. */
    kernel[0] = clCreateKernel(program[0], KERNEL_POS_NAME, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);

    /* Create the kernel from the program. */
    kernel[1] = clCreateKernel(program[1], KERNEL_NOR_NAME, &errcode_ret);
    CHECK_ERROR_CODE(errcode_ret);

    /* For setting */
    REST_LENGTH_HORIZ = CLOTH_SIZE_X / (NUM_PARTICLES_X - 1);
    REST_LENGTH_VERT = CLOTH_SIZE_Y / (NUM_PARTICLES_Y - 1);
    REST_LENGTH_DIAG = sqrtf(REST_LENGTH_HORIZ*REST_LENGTH_HORIZ + REST_LENGTH_VERT * REST_LENGTH_VERT);

    return true;
}

void FinalizeOpenCL(void) {
    cl_int errcode_ret;

    clFlush(cmd_queue);
    clFinish(cmd_queue);
    errcode_ret = clReleaseKernel(kernel[0]);
    errcode_ret = clReleaseKernel(kernel[1]);
    errcode_ret = clReleaseProgram(program[0]);
    errcode_ret = clReleaseProgram(program[1]);
    errcode_ret = clReleaseMemObject(buf_pos[0]);
    errcode_ret = clReleaseMemObject(buf_pos[1]);
    errcode_ret = clReleaseMemObject(buf_vel[0]);
    errcode_ret = clReleaseMemObject(buf_vel[1]);
    errcode_ret = clReleaseMemObject(buf_normal);
    errcode_ret = clReleaseContext(context);
}

/******************************************************************************************************/

void InitializeGLEW(char *program_name, char messages[][256], int n_message_lines) {
    GLenum error;

    fprintf(stdout, "**************************************************************\n\n");
    fprintf(stdout, "  PROGRAM NAME: %s\n\n", program_name);
    fprintf(stdout, "    This program was coded for CSE3170 students\n");
    fprintf(stdout, "      of Dept. of Comp. Sci. & Eng., Sogang University.\n\n");

    for (int i = 0; i < n_message_lines; i++)
        fprintf(stdout, "%s\n", messages[i]);
    fprintf(stdout, "\n**************************************************************\n\n");

    glewExperimental = GL_TRUE;

    error = glewInit();
    if (error != GLEW_OK) {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(error));
        exit(-1);
    }
    fprintf(stdout, "*********************************************************\n");
    fprintf(stdout, " - GLEW version supported: %s\n", glewGetString(GLEW_VERSION));
    fprintf(stdout, " - OpenGL renderer: %s\n", glGetString(GL_RENDERER));
    fprintf(stdout, " - OpenGL version supported: %s\n", glGetString(GL_VERSION));
    fprintf(stdout, "*********************************************************\n\n");
}

#define N_MESSAGE_LINES 1
bool InitializeOpenGL(int argc, char* argv[]) {
    char program_name[64] = "Sogang CSEG437/5437 Physically-based Cloth Simulation";
    char messages[N_MESSAGE_LINES][256] = { "    - Keys used: 'a', 'f', 'l', 'ESC'" };

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitContextVersion(4, 0);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    glutCreateWindow(program_name);

    InitializeGLEW(program_name, messages, N_MESSAGE_LINES);

    return true;
}

void FinalizeOpenGL(void) {
    glDeleteVertexArrays(1, &cloth_VAO);
    glDeleteBuffers(7, cloth_BOs);

    glDeleteTextures(N_TEXTURES_USED, texture_names);
}

void display(void) {
    // run OpenCL kernel
    printf("here..?\n");

    cl_int errcode_ret;
    float compute_time;
    size_t buffer_size = NUM_PARTICLES_X * NUM_PARTICLES_Y;

    size_t global_work_size[3] = { NUM_PARTICLES_X, NUM_PARTICLES_Y, 0 };
    size_t local_work_size[3] = { WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 0 };

    glFlush();
    glFinish();

    errcode_ret = clEnqueueAcquireGLObjects(cmd_queue, 2, buf_pos, 0, nullptr, nullptr);
    CHECK_ERROR_CODE(errcode_ret);
    errcode_ret = clEnqueueAcquireGLObjects(cmd_queue, 2, buf_vel, 0, nullptr, nullptr);
    CHECK_ERROR_CODE(errcode_ret);
    errcode_ret = clEnqueueAcquireGLObjects(cmd_queue, 1, &buf_normal, 0, nullptr, nullptr);
    CHECK_ERROR_CODE(errcode_ret);

    if(use_cpu)
    {
        copy_to_host(cmd_queue, &buf_pos[0], buffer_size * 4, &buf_vel[0], buffer_size * 4);
        cloth_position_host(GRAVITY, // float3,
                            PARTICLE_MASS,
                            PARTICLE_INV_MASS,
                            SPRING_K,
                            REST_LENGTH_HORIZ,
                            REST_LENGTH_VERT,
                            REST_LENGTH_DIAG,
                            DELTA_T,
                            DAMPING_CONST);
        copy_to_device(cmd_queue, &buf_pos[1], buffer_size * 4, &buf_vel[1], buffer_size * 4 );
    }
    else
    {
        int read_buf = 0;

        CHECK_TIME_START;
        for (int i = 0; i < NUM_ITER; i++) {
            errcode_ret  = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &buf_pos[read_buf]);
            errcode_ret |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &buf_pos[1-read_buf]);
            errcode_ret |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &buf_vel[read_buf]);
            errcode_ret |= clSetKernelArg(kernel[0], 3, sizeof(cl_mem), &buf_vel[1-read_buf]);
            errcode_ret |= clSetKernelArg(kernel[0], 4, 4 * (WORKGROUP_SIZE_X + 2) * (WORKGROUP_SIZE_Y + 2) * sizeof(float), NULL);
            errcode_ret |= clSetKernelArg(kernel[0], 5, 4 * sizeof(float), GRAVITY);
            errcode_ret |= clSetKernelArg(kernel[0], 6, sizeof(float), &PARTICLE_MASS);
            errcode_ret |= clSetKernelArg(kernel[0], 7, sizeof(float), &PARTICLE_INV_MASS);
            errcode_ret |= clSetKernelArg(kernel[0], 8, sizeof(float), &SPRING_K);
            errcode_ret |= clSetKernelArg(kernel[0], 9, sizeof(float), &REST_LENGTH_HORIZ);
            errcode_ret |= clSetKernelArg(kernel[0], 10, sizeof(float), &REST_LENGTH_VERT);
            errcode_ret |= clSetKernelArg(kernel[0], 11, sizeof(float), &REST_LENGTH_DIAG);
            errcode_ret |= clSetKernelArg(kernel[0], 12, sizeof(float), &DELTA_T);
            errcode_ret |= clSetKernelArg(kernel[0], 13, sizeof(float), &DAMPING_CONST);
            CHECK_ERROR_CODE(errcode_ret);
            read_buf = 1 - read_buf;

            errcode_ret = clEnqueueNDRangeKernel(cmd_queue, kernel[0], 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
            CHECK_ERROR_CODE(errcode_ret);
            clFinish(cmd_queue);
        }
    }
    errcode_ret  = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &buf_pos[0]);
    errcode_ret |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &buf_normal);
    errcode_ret |= clSetKernelArg(kernel[1], 2, 4 * (WORKGROUP_SIZE_X + 2) * (WORKGROUP_SIZE_Y + 2) * sizeof(float), NULL);
    CHECK_ERROR_CODE(errcode_ret);
    errcode_ret = clEnqueueNDRangeKernel(cmd_queue, kernel[1], 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
    CHECK_ERROR_CODE(errcode_ret);

    clFinish(cmd_queue);
    CHECK_TIME_END(compute_time);

    clFlush(cmd_queue);
    clFinish(cmd_queue);
    errcode_ret = clEnqueueReleaseGLObjects(cmd_queue, 2, buf_pos, 0, nullptr, nullptr);
    CHECK_ERROR_CODE(errcode_ret);
    errcode_ret = clEnqueueReleaseGLObjects(cmd_queue, 2, buf_vel, 0, nullptr, nullptr);
    CHECK_ERROR_CODE(errcode_ret);
    errcode_ret = clEnqueueReleaseGLObjects(cmd_queue, 1, &buf_normal, 0, nullptr, nullptr);
    CHECK_ERROR_CODE(errcode_ret);

    fprintf(stdout, "     * Time by CL kernel = %.3fms\n\n", compute_time);

        // run OpenGL
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(h_ShaderProgram_Phong);

    glUniform1i(loc_texture, TEXTURE_ID_CLOTH);
    ViewMatrix = glm::lookAt(glm::vec3(3, 2, 5), glm::vec3(2, 1, 0), glm::vec3(0, 1, 0));
    ModelViewMatrix = ViewMatrix * glm::mat4(1.0f);
    ModelViewProjectionMatrix = ProjectionMatrix * ModelViewMatrix;
    ModelViewMatrixInvTrans = glm::inverseTranspose(glm::mat3(ModelViewMatrix));

    glUniformMatrix4fv(loc_ModelViewProjectionMatrix_TXPS, 1, GL_FALSE, &ModelViewProjectionMatrix[0][0]);
    glUniformMatrix4fv(loc_ModelViewMatrix_TXPS, 1, GL_FALSE, &ModelViewMatrix[0][0]);
    glUniformMatrix3fv(loc_ModelViewMatrixInvTrans_TXPS, 1, GL_FALSE, &ModelViewMatrixInvTrans[0][0]);

    glBindVertexArray(cloth_VAO);
    glDrawElements(GL_TRIANGLE_STRIP, 2 * NUM_PARTICLES_X * NUM_PARTICLES_Y, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glUseProgram(0);

    glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
	case 'f':
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glutPostRedisplay();
		break;
	case 'l':
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glutPostRedisplay();
		break;
    case 27: // ESC key
        glutLeaveMainLoop(); // Incur destuction callback for cleanups
        break;
    }
}

void reshape(int width, int height) {
    float aspect_ratio;
    glViewport(0, 0, width, height);

    aspect_ratio = (float)width / height;
    ProjectionMatrix = glm::perspective(glm::radians(50.0f), aspect_ratio, 1.0f, 100.0f);

    glutPostRedisplay();
}

// unsigned int dummy_var = 0;
void timer_scene(int timestamp_scene) {

    glutPostRedisplay();
	/*
	if (timestamp_scene % 60 == 0) {
		printf("--- Time stamp = %d\n", timestamp_scene);
		scanf("%d", &dummy_var);
	}
	*/
    glutTimerFunc(5, timer_scene, (timestamp_scene + 1) % INT32_MAX);
}

void cleanup(void) {
    FinalizeOpenCL();
    FinalizeOpenGL();
}

void RegisterCallbacks(void) {
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutTimerFunc(5, timer_scene, 0);
    glutCloseFunc(cleanup);
}

void PrepareShaderProgram(void) {
    int i;
    char string[256];
    ShaderInfo shader_info_phong[3] = {
        { GL_VERTEX_SHADER, "shaders/phong.vert" },
        { GL_FRAGMENT_SHADER, "shaders/phong.frag" },
        { GL_NONE, NULL }
    };

    h_ShaderProgram_Phong = LoadShaders(shader_info_phong);
    loc_ModelViewProjectionMatrix_TXPS = glGetUniformLocation(h_ShaderProgram_Phong, "u_ModelViewProjectionMatrix");
    loc_ModelViewMatrix_TXPS = glGetUniformLocation(h_ShaderProgram_Phong, "u_ModelViewMatrix");
    loc_ModelViewMatrixInvTrans_TXPS = glGetUniformLocation(h_ShaderProgram_Phong, "u_ModelViewMatrixInvTrans");

    loc_global_ambient_color = glGetUniformLocation(h_ShaderProgram_Phong, "u_global_ambient_color");
    for (i = 0; i < NUMBER_OF_LIGHT_SUPPORTED; i++) {
        sprintf(string, "u_light[%d].light_on", i);
        loc_light[i].light_on = glGetUniformLocation(h_ShaderProgram_Phong, string);
        sprintf(string, "u_light[%d].position", i);
        loc_light[i].position = glGetUniformLocation(h_ShaderProgram_Phong, string);
        sprintf(string, "u_light[%d].ambient_color", i);
        loc_light[i].ambient_color = glGetUniformLocation(h_ShaderProgram_Phong, string);
        sprintf(string, "u_light[%d].diffuse_color", i);
        loc_light[i].diffuse_color = glGetUniformLocation(h_ShaderProgram_Phong, string);
        sprintf(string, "u_light[%d].specular_color", i);
        loc_light[i].specular_color = glGetUniformLocation(h_ShaderProgram_Phong, string);
        sprintf(string, "u_light[%d].spot_direction", i);
        loc_light[i].spot_direction = glGetUniformLocation(h_ShaderProgram_Phong, string);
        sprintf(string, "u_light[%d].spot_exponent", i);
        loc_light[i].spot_exponent = glGetUniformLocation(h_ShaderProgram_Phong, string);
        sprintf(string, "u_light[%d].spot_cutoff_angle", i);
        loc_light[i].spot_cutoff_angle = glGetUniformLocation(h_ShaderProgram_Phong, string);
        sprintf(string, "u_light[%d].light_attenuation_factors", i);
        loc_light[i].light_attenuation_factors = glGetUniformLocation(h_ShaderProgram_Phong, string);
    }

    loc_material.ambient_color = glGetUniformLocation(h_ShaderProgram_Phong, "u_material.ambient_color");
    loc_material.diffuse_color = glGetUniformLocation(h_ShaderProgram_Phong, "u_material.diffuse_color");
    loc_material.specular_color = glGetUniformLocation(h_ShaderProgram_Phong, "u_material.specular_color");
    loc_material.emissive_color = glGetUniformLocation(h_ShaderProgram_Phong, "u_material.emissive_color");
    loc_material.specular_exponent = glGetUniformLocation(h_ShaderProgram_Phong, "u_material.specular_exponent");

    loc_texture = glGetUniformLocation(h_ShaderProgram_Phong, "u_base_texture");
}

void initialize_lights_and_material(void) { // follow OpenGL conventions for initialization
    int i;

    glUseProgram(h_ShaderProgram_Phong);

    glUniform4f(loc_global_ambient_color, 0.115f, 0.115f, 0.115f, 1.0f);
    for (i = 0; i < NUMBER_OF_LIGHT_SUPPORTED; i++) {
        glUniform1i(loc_light[i].light_on, 0); // turn off all lights initially
        glUniform4f(loc_light[i].position, 0.0f, 0.0f, 1.0f, 0.0f);
        glUniform4f(loc_light[i].ambient_color, 0.0f, 0.0f, 0.0f, 1.0f);
        if (i == 0) {
            glUniform4f(loc_light[i].diffuse_color, 1.0f, 1.0f, 1.0f, 1.0f);
            glUniform4f(loc_light[i].specular_color, 1.0f, 1.0f, 1.0f, 1.0f);
        }
        else {
            glUniform4f(loc_light[i].diffuse_color, 0.0f, 0.0f, 0.0f, 1.0f);
            glUniform4f(loc_light[i].specular_color, 0.0f, 0.0f, 0.0f, 1.0f);
        }
        glUniform3f(loc_light[i].spot_direction, 0.0f, 0.0f, -1.0f);
        glUniform1f(loc_light[i].spot_exponent, 0.0f); // [0.0, 128.0]
        glUniform1f(loc_light[i].spot_cutoff_angle, 180.0f); // [0.0, 90.0] or 180.0 (180.0 for no spot light effect)
        glUniform4f(loc_light[i].light_attenuation_factors, 1.0f, 0.0f, 0.0f, 0.0f); // .w != 0.0f for no ligth attenuation
    }

    glUniform4f(loc_material.ambient_color, 0.2f, 0.2f, 0.2f, 1.0f);
    glUniform4f(loc_material.diffuse_color, 0.8f, 0.8f, 0.8f, 1.0f);
    glUniform4f(loc_material.specular_color, 0.0f, 0.0f, 0.0f, 1.0f);
    glUniform4f(loc_material.emissive_color, 0.0f, 0.0f, 0.0f, 1.0f);
    glUniform1f(loc_material.specular_exponent, 0.0f); // [0.0, 128.0]

    glUseProgram(0);
}

void InitializeOpenGLSetting(void) {
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_MULTISAMPLE);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    ViewMatrix = glm::lookAt(glm::vec3(1000.0f, 1000.0f, 1000.0f), glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f));

    initialize_lights_and_material();

    glGenTextures(N_TEXTURES_USED, texture_names);

    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(PRIM_RESTART);
}

void glTexImage2D_from_file(const char *filename) {
    FREE_IMAGE_FORMAT tx_file_format;
    int tx_bits_per_pixel;
    FIBITMAP *tx_pixmap, *tx_pixmap_32;

    int width, height;
    GLvoid *data;

    tx_file_format = FreeImage_GetFileType(filename, 0);
    // assume everything is fine with reading texture from file: no error checking
    tx_pixmap = FreeImage_Load(tx_file_format, filename);
    tx_bits_per_pixel = FreeImage_GetBPP(tx_pixmap);

    fprintf(stdout, " * A %d-bit texture was read from %s.\n", tx_bits_per_pixel, filename);
    if (tx_bits_per_pixel == 32)
        tx_pixmap_32 = tx_pixmap;
    else {
        fprintf(stdout, " * Converting texture from %d bits to 32 bits...\n", tx_bits_per_pixel);
        tx_pixmap_32 = FreeImage_ConvertTo32Bits(tx_pixmap);
    }

    width = FreeImage_GetWidth(tx_pixmap_32);
    height = FreeImage_GetHeight(tx_pixmap_32);
    data = FreeImage_GetBits(tx_pixmap_32);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, data);
    fprintf(stdout, " * Loaded %dx%d RGBA texture into graphics memory.\n\n", width, height);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    FreeImage_Unload(tx_pixmap_32);
    if (tx_bits_per_pixel != 32)
        FreeImage_Unload(tx_pixmap);
}

Material_Parameters material_cloth;
void prepare_cloth(void) { // Draw coordinate axes.
                           // Initialize vertex buffer object.
    material_cloth.ambient_color[0] = 0.2f;
    material_cloth.ambient_color[1] = 0.2f;
    material_cloth.ambient_color[2] = 0.2f;
    material_cloth.ambient_color[3] = 1.0f;

    material_cloth.diffuse_color[0] = 0.8f;
    material_cloth.diffuse_color[1] = 0.8f;
    material_cloth.diffuse_color[2] = 0.8f;
    material_cloth.diffuse_color[3] = 1.0f;

    material_cloth.specular_color[0] = 0.2f;
    material_cloth.specular_color[1] = 0.2f;
    material_cloth.specular_color[2] = 0.2f;
    material_cloth.specular_color[3] = 1.0f;

    material_cloth.specular_exponent = 80.0f;

    material_cloth.emissive_color[0] = 0.0f;
    material_cloth.emissive_color[1] = 0.0f;
    material_cloth.emissive_color[2] = 0.0f;
    material_cloth.emissive_color[3] = 1.0f;
    glHint(GL_GENERATE_MIPMAP_HINT, GL_NICEST);

    glActiveTexture(GL_TEXTURE0 + TEXTURE_ID_CLOTH);
    glBindTexture(GL_TEXTURE_2D, texture_names[TEXTURE_ID_CLOTH]);

    glTexImage2D_from_file("flag.png");

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}

void set_up_scene_lights(void) {
    // point_light_EC: use light 0
    light[0].light_on = 1;
    light[0].position[0] = 0.0f; light[0].position[1] = 0.0f; 	// point light position in EC
    light[0].position[2] = 0.0f; light[0].position[3] = 1.0f;

    light[0].ambient_color[0] = 0.13f; light[0].ambient_color[1] = 0.13f;
    light[0].ambient_color[2] = 0.13f; light[0].ambient_color[3] = 1.0f;

    light[0].diffuse_color[0] = 0.5f; light[0].diffuse_color[1] = 0.5f;
    light[0].diffuse_color[2] = 0.5f; light[0].diffuse_color[3] = 1.0f;

    light[0].specular_color[0] = 0.8f; light[0].specular_color[1] = 0.8f;
    light[0].specular_color[2] = 0.8f; light[0].specular_color[3] = 1.0f;

    glUseProgram(h_ShaderProgram_Phong);
    glUniform1i(loc_light[0].light_on, light[0].light_on);
    glUniform4fv(loc_light[0].position, 1, light[0].position);
    glUniform4fv(loc_light[0].ambient_color, 1, light[0].ambient_color);
    glUniform4fv(loc_light[0].diffuse_color, 1, light[0].diffuse_color);
    glUniform4fv(loc_light[0].specular_color, 1, light[0].specular_color);

    glUseProgram(0);
}

void PrepareScene(void) {
    prepare_cloth();
    set_up_scene_lights();
}

void InitializeRenderer() {
    InitializeBuffers();
    RegisterCallbacks();
    PrepareShaderProgram();
    InitializeOpenGLSetting();
    PrepareScene();
}

/******************************************************************************************************/


int main(int argc, char* argv[]) {
	printf("\n^^^ Input the number of subintervals that subdivide the 1/60-second interval: ");
	scanf("%d", &NUM_ITER);
	printf("^^^ NUN_ITER = %d, DELTA_T = %e \n\n", NUM_ITER, DELTA_T);
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));

    if (!InitializeOpenGL(argc, argv)) {
        fprintf(stdout, "Error : OpenGL is not initialized.\n");
        return -1;
    }
    if (!InitializeOpenCL()) {
        fprintf(stdout, "Error : OpenCL is not initialized.\n");
        FinalizeOpenGL();
        return -1;
    }

    InitializeRenderer();

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutMainLoop();
}
