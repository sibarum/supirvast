package dev.supirvast.vastir.preview;

import dev.supirvast.vastir.tools.Fullscreen;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.glfw.GLFWVulkan.glfwCreateWindowSurface;
import static org.lwjgl.glfw.GLFWVulkan.glfwGetRequiredInstanceExtensions;
import static org.lwjgl.glfw.GLFWVulkan.glfwVulkanSupported;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.vulkan.KHRSurface.*;
import static org.lwjgl.vulkan.KHRSwapchain.*;
import static org.lwjgl.vulkan.VK13.*;

/**
 * A minimal windowed Vulkan host for <em>fullscreen</em> shader passes: it owns the GLFW window, swapchain, and
 * present loop, binds a vertex + fragment SPIR-V pair into a pipeline with an <strong>empty vertex input</strong>
 * (no buffers, no attributes), and draws three vertices per frame — the fullscreen-triangle pattern authored by
 * {@link dev.supirvast.vastir.tools.Fullscreen}. Clear + one {@code vkCmdDraw(3,1,0,0)} into the swapchain image.
 *
 * <p>This is the graphics counterpart to {@code VulkanCompute} in vastir-tools — a reusable host that any project
 * (not just a front end) can drive. It is deliberately smaller than {@link PreviewApp}: no depth buffer, no vertex
 * buffers, no textures, no push constants, no MVP — the things a screen-space effect does not need. Later phases
 * grow it (a per-frame push-constant seam for resolution/time, then a frame callback).
 *
 * <p>Threading: GLFW init, window creation, and event polling all happen on the constructing/looping thread, and
 * {@link #close()} calls {@code glfwTerminate} — so this instance assumes it owns GLFW for the process, exactly as
 * {@link PreviewApp} does. Windows-only, Vulkan 1.3, matching the rest of the previewer.
 */
public final class WindowedVulkanContext implements AutoCloseable {

    private final String title;
    private final int requestedWidth;
    private final int requestedHeight;
    private final byte[] vertexSpirv;
    private final String vertexEntry;
    private final byte[] fragmentSpirv;
    private final String fragmentEntry;

    private long window;
    private VkInstance instance;
    private long surface;
    private VkPhysicalDevice physical;
    private VkDevice device;
    private int queueFamily;
    private VkQueue queue;

    private long swapchain;
    private int swapchainFormat;
    private int extentWidth;
    private int extentHeight;
    private long[] swapchainImages = new long[0];
    private long[] imageViews = new long[0];
    private long[] framebuffers = new long[0];
    private long renderPass;

    private long vertexShaderModule;
    private long fragmentShaderModule;
    private long pipelineLayout;
    private long graphicsPipeline;

    private long commandPool;
    private VkCommandBuffer commandBuffer;

    private long imageAvailable;
    private long renderFinished;
    private long inFlight;

    public WindowedVulkanContext(String title, int width, int height,
                                 byte[] vertexSpirv, String vertexEntry,
                                 byte[] fragmentSpirv, String fragmentEntry) {
        this.title = title;
        this.requestedWidth = width;
        this.requestedHeight = height;
        this.vertexSpirv = vertexSpirv.clone();
        this.vertexEntry = vertexEntry;
        this.fragmentSpirv = fragmentSpirv.clone();
        this.fragmentEntry = fragmentEntry;
        // A driver can expose hundreds of device extensions; LWJGL enumerates them all in one MemoryStack malloc
        // during VkInstance construction, which overflows the default 64 KB thread stack. Give it room. Must be set
        // before this thread's first stackPush.
        Configuration.STACK_SIZE.set(1024);
        initWindow();
        initVulkan();
    }

    /**
     * Renders until the window is closed, or until {@code maxFrames} have been presented (whichever comes first).
     * A non-positive {@code maxFrames} means render until the user closes the window; a small positive value is a
     * smoke test that verifies the whole present path on real hardware without a human in the loop.
     */
    public void run(int maxFrames) {
        int rendered = 0;
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
            rendered++;
            if (maxFrames > 0 && rendered >= maxFrames) {
                break;
            }
        }
        vkDeviceWaitIdle(device);
    }

    private void initWindow() {
        if (!glfwInit()) {
            throw new IllegalStateException("failed to initialize GLFW");
        }
        if (!glfwVulkanSupported()) {
            throw new IllegalStateException("GLFW reports Vulkan is not supported on this machine");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // no OpenGL context; this is a Vulkan window
        window = glfwCreateWindow(requestedWidth, requestedHeight, title, NULL, NULL);
        if (window == NULL) {
            throw new IllegalStateException("failed to create the GLFW window");
        }
    }

    private void initVulkan() {
        try (MemoryStack stack = stackPush()) {
            createInstance(stack);
            createSurface(stack);
            pickPhysicalDevice(stack);
            createDevice(stack);
            createSwapchain(stack);
            createRenderPass(stack);
            createFramebuffers(stack);
            createCommandPoolAndBuffer(stack);
            createGraphicsPipeline(stack);
            createSyncObjects(stack);
        }
    }

    private void createInstance(MemoryStack stack) {
        PointerBuffer glfwExtensions = glfwGetRequiredInstanceExtensions();
        if (glfwExtensions == null) {
            throw new IllegalStateException("GLFW could not enumerate the required Vulkan instance extensions");
        }
        VkApplicationInfo app = VkApplicationInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_APPLICATION_INFO)
                .pApplicationName(stack.UTF8(title))
                .apiVersion(VK_API_VERSION_1_3);
        VkInstanceCreateInfo info = VkInstanceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
                .pApplicationInfo(app)
                .ppEnabledExtensionNames(glfwExtensions);
        PointerBuffer pInstance = stack.mallocPointer(1);
        check(vkCreateInstance(info, null, pInstance), "vkCreateInstance");
        instance = new VkInstance(pInstance.get(0), info);
    }

    private void createSurface(MemoryStack stack) {
        LongBuffer pSurface = stack.mallocLong(1);
        check(glfwCreateWindowSurface(instance, window, null, pSurface), "glfwCreateWindowSurface");
        surface = pSurface.get(0);
    }

    private void pickPhysicalDevice(MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        check(vkEnumeratePhysicalDevices(instance, count, null), "vkEnumeratePhysicalDevices");
        if (count.get(0) == 0) {
            throw new IllegalStateException("no Vulkan physical devices found");
        }
        PointerBuffer devices = stack.mallocPointer(count.get(0));
        check(vkEnumeratePhysicalDevices(instance, count, devices), "vkEnumeratePhysicalDevices");
        for (int i = 0; i < devices.capacity(); i++) {
            VkPhysicalDevice candidate = new VkPhysicalDevice(devices.get(i), instance);
            int family = findGraphicsPresentQueueFamily(candidate, stack);
            if (family >= 0) {
                physical = candidate;
                queueFamily = family;
                return;
            }
        }
        throw new IllegalStateException("no device with a graphics + present queue family for this surface");
    }

    private int findGraphicsPresentQueueFamily(VkPhysicalDevice candidate, MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, count, null);
        VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.malloc(count.get(0), stack);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, count, families);
        IntBuffer presentSupport = stack.mallocInt(1);
        for (int i = 0; i < families.capacity(); i++) {
            boolean graphics = (families.get(i).queueFlags() & VK_QUEUE_GRAPHICS_BIT) != 0;
            check(vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface, presentSupport),
                    "vkGetPhysicalDeviceSurfaceSupportKHR");
            if (graphics && presentSupport.get(0) == VK_TRUE) {
                return i;
            }
        }
        return -1;
    }

    private void createDevice(MemoryStack stack) {
        VkDeviceQueueCreateInfo.Buffer queues = VkDeviceQueueCreateInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
                .queueFamilyIndex(queueFamily)
                .pQueuePriorities(stack.floats(1.0f));
        VkDeviceCreateInfo info = VkDeviceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
                .pQueueCreateInfos(queues)
                .ppEnabledExtensionNames(stack.pointers(stack.UTF8(VK_KHR_SWAPCHAIN_EXTENSION_NAME)));
        PointerBuffer pDevice = stack.mallocPointer(1);
        check(vkCreateDevice(physical, info, null, pDevice), "vkCreateDevice");
        device = new VkDevice(pDevice.get(0), physical, info);

        PointerBuffer pQueue = stack.mallocPointer(1);
        vkGetDeviceQueue(device, queueFamily, 0, pQueue);
        queue = new VkQueue(pQueue.get(0), device);
    }

    private void createSwapchain(MemoryStack stack) {
        VkSurfaceCapabilitiesKHR caps = VkSurfaceCapabilitiesKHR.malloc(stack);
        check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, caps),
                "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

        VkSurfaceFormatKHR.Buffer formats = surfaceFormats(stack);
        VkSurfaceFormatKHR chosen = chooseSurfaceFormat(formats);
        swapchainFormat = chosen.format();
        chooseExtent(caps);

        int imageCount = caps.minImageCount() + 1;
        if (caps.maxImageCount() > 0 && imageCount > caps.maxImageCount()) {
            imageCount = caps.maxImageCount();
        }

        VkSwapchainCreateInfoKHR info = VkSwapchainCreateInfoKHR.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR)
                .surface(surface)
                .minImageCount(imageCount)
                .imageFormat(swapchainFormat)
                .imageColorSpace(chosen.colorSpace())
                .imageArrayLayers(1)
                .imageUsage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
                .imageSharingMode(VK_SHARING_MODE_EXCLUSIVE)
                .preTransform(caps.currentTransform())
                .compositeAlpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                .presentMode(VK_PRESENT_MODE_FIFO_KHR)
                .clipped(true)
                .oldSwapchain(VK_NULL_HANDLE);
        info.imageExtent().width(extentWidth).height(extentHeight);

        LongBuffer pSwapchain = stack.mallocLong(1);
        check(vkCreateSwapchainKHR(device, info, null, pSwapchain), "vkCreateSwapchainKHR");
        swapchain = pSwapchain.get(0);

        swapchainImages = swapchainImages(stack);
        imageViews = new long[swapchainImages.length];
        for (int i = 0; i < swapchainImages.length; i++) {
            imageViews[i] = createColorImageView(swapchainImages[i], stack);
        }
    }

    private VkSurfaceFormatKHR.Buffer surfaceFormats(MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, count, null),
                "vkGetPhysicalDeviceSurfaceFormatsKHR");
        VkSurfaceFormatKHR.Buffer formats = VkSurfaceFormatKHR.malloc(count.get(0), stack);
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, count, formats),
                "vkGetPhysicalDeviceSurfaceFormatsKHR");
        return formats;
    }

    private static VkSurfaceFormatKHR chooseSurfaceFormat(VkSurfaceFormatKHR.Buffer formats) {
        for (int i = 0; i < formats.capacity(); i++) {
            VkSurfaceFormatKHR f = formats.get(i);
            boolean bgra = f.format() == VK_FORMAT_B8G8R8A8_UNORM || f.format() == VK_FORMAT_B8G8R8A8_SRGB;
            if (bgra && f.colorSpace() == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return f;
            }
        }
        return formats.get(0);
    }

    private void chooseExtent(VkSurfaceCapabilitiesKHR caps) {
        VkExtent2D current = caps.currentExtent();
        if (current.width() != 0xFFFFFFFF) {
            extentWidth = current.width();
            extentHeight = current.height();
            return;
        }
        extentWidth = clamp(requestedWidth, caps.minImageExtent().width(), caps.maxImageExtent().width());
        extentHeight = clamp(requestedHeight, caps.minImageExtent().height(), caps.maxImageExtent().height());
    }

    private long[] swapchainImages(MemoryStack stack) {
        IntBuffer count = stack.mallocInt(1);
        check(vkGetSwapchainImagesKHR(device, swapchain, count, null), "vkGetSwapchainImagesKHR");
        LongBuffer pImages = stack.mallocLong(count.get(0));
        check(vkGetSwapchainImagesKHR(device, swapchain, count, pImages), "vkGetSwapchainImagesKHR");
        long[] images = new long[count.get(0)];
        pImages.get(images);
        return images;
    }

    private long createColorImageView(long image, MemoryStack stack) {
        VkImageViewCreateInfo info = VkImageViewCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
                .image(image)
                .viewType(VK_IMAGE_VIEW_TYPE_2D)
                .format(swapchainFormat);
        info.subresourceRange()
                .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                .baseMipLevel(0).levelCount(1)
                .baseArrayLayer(0).layerCount(1);
        LongBuffer pView = stack.mallocLong(1);
        check(vkCreateImageView(device, info, null, pView), "vkCreateImageView");
        return pView.get(0);
    }

    private void createRenderPass(MemoryStack stack) {
        VkAttachmentDescription.Buffer attachments = VkAttachmentDescription.calloc(1, stack);
        attachments.get(0)   // single color attachment, cleared then presented
                .format(swapchainFormat)
                .samples(VK_SAMPLE_COUNT_1_BIT)
                .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE)
                .stencilLoadOp(VK_ATTACHMENT_LOAD_OP_DONT_CARE)
                .stencilStoreOp(VK_ATTACHMENT_STORE_OP_DONT_CARE)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED)
                .finalLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

        VkAttachmentReference.Buffer colorRef = VkAttachmentReference.calloc(1, stack)
                .attachment(0)
                .layout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        VkSubpassDescription.Buffer subpass = VkSubpassDescription.calloc(1, stack)
                .pipelineBindPoint(VK_PIPELINE_BIND_POINT_GRAPHICS)
                .colorAttachmentCount(1)
                .pColorAttachments(colorRef);

        VkSubpassDependency.Buffer dependency = VkSubpassDependency.calloc(1, stack);
        dependency.get(0)
                .srcSubpass(VK_SUBPASS_EXTERNAL)
                .dstSubpass(0)
                .srcStageMask(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
                .srcAccessMask(0)
                .dstStageMask(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
                .dstAccessMask(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

        VkRenderPassCreateInfo info = VkRenderPassCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO)
                .pAttachments(attachments)
                .pSubpasses(subpass)
                .pDependencies(dependency);
        LongBuffer pRenderPass = stack.mallocLong(1);
        check(vkCreateRenderPass(device, info, null, pRenderPass), "vkCreateRenderPass");
        renderPass = pRenderPass.get(0);
    }

    private void createFramebuffers(MemoryStack stack) {
        framebuffers = new long[imageViews.length];
        for (int i = 0; i < imageViews.length; i++) {
            VkFramebufferCreateInfo info = VkFramebufferCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO)
                    .renderPass(renderPass)
                    .pAttachments(stack.longs(imageViews[i]))
                    .width(extentWidth)
                    .height(extentHeight)
                    .layers(1);
            LongBuffer pFramebuffer = stack.mallocLong(1);
            check(vkCreateFramebuffer(device, info, null, pFramebuffer), "vkCreateFramebuffer");
            framebuffers[i] = pFramebuffer.get(0);
        }
    }

    private void createCommandPoolAndBuffer(MemoryStack stack) {
        VkCommandPoolCreateInfo poolInfo = VkCommandPoolCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
                .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
                .queueFamilyIndex(queueFamily);
        LongBuffer pPool = stack.mallocLong(1);
        check(vkCreateCommandPool(device, poolInfo, null, pPool), "vkCreateCommandPool");
        commandPool = pPool.get(0);

        VkCommandBufferAllocateInfo allocInfo = VkCommandBufferAllocateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO)
                .commandPool(commandPool)
                .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                .commandBufferCount(1);
        PointerBuffer pBuffer = stack.mallocPointer(1);
        check(vkAllocateCommandBuffers(device, allocInfo, pBuffer), "vkAllocateCommandBuffers");
        commandBuffer = new VkCommandBuffer(pBuffer.get(0), device);
    }

    private void createGraphicsPipeline(MemoryStack stack) {
        vertexShaderModule = createShaderModule(vertexSpirv, stack);
        fragmentShaderModule = createShaderModule(fragmentSpirv, stack);

        VkPipelineShaderStageCreateInfo.Buffer stages = VkPipelineShaderStageCreateInfo.calloc(2, stack);
        stages.get(0)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                .stage(VK_SHADER_STAGE_VERTEX_BIT)
                .module(vertexShaderModule)
                .pName(stack.UTF8(vertexEntry));
        stages.get(1)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO)
                .stage(VK_SHADER_STAGE_FRAGMENT_BIT)
                .module(fragmentShaderModule)
                .pName(stack.UTF8(fragmentEntry));

        // Empty vertex input: the fullscreen triangle is generated from gl_VertexIndex — no buffers, no attributes.
        VkPipelineVertexInputStateCreateInfo vertexInput = VkPipelineVertexInputStateCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);

        VkPipelineInputAssemblyStateCreateInfo inputAssembly = VkPipelineInputAssemblyStateCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO)
                .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        VkViewport.Buffer viewport = VkViewport.calloc(1, stack)
                .x(0).y(0).width(extentWidth).height(extentHeight).minDepth(0).maxDepth(1);
        VkRect2D.Buffer scissor = VkRect2D.calloc(1, stack);
        scissor.get(0).offset().set(0, 0);
        scissor.get(0).extent().width(extentWidth).height(extentHeight);
        VkPipelineViewportStateCreateInfo viewportState = VkPipelineViewportStateCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO)
                .pViewports(viewport)
                .pScissors(scissor);

        VkPipelineRasterizationStateCreateInfo rasterizer = VkPipelineRasterizationStateCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO)
                .polygonMode(VK_POLYGON_MODE_FILL)
                .cullMode(VK_CULL_MODE_NONE)
                .frontFace(VK_FRONT_FACE_COUNTER_CLOCKWISE)
                .lineWidth(1.0f);

        VkPipelineMultisampleStateCreateInfo multisample = VkPipelineMultisampleStateCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO)
                .rasterizationSamples(VK_SAMPLE_COUNT_1_BIT);

        VkPipelineColorBlendAttachmentState.Buffer blendAttachment =
                VkPipelineColorBlendAttachmentState.calloc(1, stack)
                        .colorWriteMask(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT)
                        .blendEnable(false);
        VkPipelineColorBlendStateCreateInfo colorBlend = VkPipelineColorBlendStateCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO)
                .pAttachments(blendAttachment);

        // A fragment push-constant range for the standard per-frame uniforms (resolution, time). Always
        // declared — a shader that reads them matches this block; one that ignores them simply doesn't.
        VkPipelineLayoutCreateInfo layoutInfo = VkPipelineLayoutCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
                .pPushConstantRanges(VkPushConstantRange.calloc(1, stack)
                        .stageFlags(VK_SHADER_STAGE_FRAGMENT_BIT)
                        .offset(0).size(Fullscreen.STANDARD_UNIFORM_BYTES));
        LongBuffer pLayout = stack.mallocLong(1);
        check(vkCreatePipelineLayout(device, layoutInfo, null, pLayout), "vkCreatePipelineLayout");
        pipelineLayout = pLayout.get(0);

        VkGraphicsPipelineCreateInfo.Buffer info = VkGraphicsPipelineCreateInfo.calloc(1, stack)
                .sType(VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO)
                .pStages(stages)
                .pVertexInputState(vertexInput)
                .pInputAssemblyState(inputAssembly)
                .pViewportState(viewportState)
                .pRasterizationState(rasterizer)
                .pMultisampleState(multisample)
                .pColorBlendState(colorBlend)
                .layout(pipelineLayout)
                .renderPass(renderPass)
                .subpass(0);
        LongBuffer pPipeline = stack.mallocLong(1);
        check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, info, null, pPipeline),
                "vkCreateGraphicsPipelines");
        graphicsPipeline = pPipeline.get(0);
    }

    private long createShaderModule(byte[] spirv, MemoryStack stack) {
        ByteBuffer code = MemoryUtil.memAlloc(spirv.length).put(spirv);
        code.flip();
        try {
            VkShaderModuleCreateInfo info = VkShaderModuleCreateInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)
                    .pCode(code);
            LongBuffer pModule = stack.mallocLong(1);
            check(vkCreateShaderModule(device, info, null, pModule), "vkCreateShaderModule");
            return pModule.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }

    private void createSyncObjects(MemoryStack stack) {
        VkSemaphoreCreateInfo semInfo = VkSemaphoreCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
        VkFenceCreateInfo fenceInfo = VkFenceCreateInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
                .flags(VK_FENCE_CREATE_SIGNALED_BIT);
        LongBuffer p = stack.mallocLong(1);
        check(vkCreateSemaphore(device, semInfo, null, p), "vkCreateSemaphore");
        imageAvailable = p.get(0);
        check(vkCreateSemaphore(device, semInfo, null, p), "vkCreateSemaphore");
        renderFinished = p.get(0);
        check(vkCreateFence(device, fenceInfo, null, p), "vkCreateFence");
        inFlight = p.get(0);
    }

    /** Clear + bind the fullscreen pipeline + draw three vertices (the triangle is generated in the shader). */
    private void recordDraw(long framebuffer, MemoryStack stack) {
        VkCommandBufferBeginInfo begin = VkCommandBufferBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        check(vkBeginCommandBuffer(commandBuffer, begin), "vkBeginCommandBuffer");

        VkClearValue.Buffer clears = VkClearValue.calloc(1, stack);
        clears.get(0).color().float32(0, 0.0f).float32(1, 0.0f).float32(2, 0.0f).float32(3, 1.0f);

        VkRect2D area = VkRect2D.calloc(stack);
        area.offset().set(0, 0);
        area.extent().width(extentWidth).height(extentHeight);

        VkRenderPassBeginInfo rpBegin = VkRenderPassBeginInfo.calloc(stack)
                .sType(VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO)
                .renderPass(renderPass)
                .framebuffer(framebuffer)
                .renderArea(area)
                .pClearValues(clears);
        vkCmdBeginRenderPass(commandBuffer, rpBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, standardUniforms(stack));
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);   // fullscreen triangle, no vertex buffer
        vkCmdEndRenderPass(commandBuffer);
        check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    }

    /** The per-frame uniform block the fragment stage reads: {@code vec2 resolution} (0), {@code float time} (8). */
    private ByteBuffer standardUniforms(MemoryStack stack) {
        ByteBuffer buffer = stack.malloc(Fullscreen.STANDARD_UNIFORM_BYTES);
        buffer.putFloat(0, (float) extentWidth);
        buffer.putFloat(4, (float) extentHeight);
        buffer.putFloat(8, (float) glfwGetTime());   // seconds since GLFW init — drives animation
        return buffer;
    }

    private void drawFrame() {
        try (MemoryStack stack = stackPush()) {
            check(vkWaitForFences(device, stack.longs(inFlight), true, Long.MAX_VALUE), "vkWaitForFences");
            check(vkResetFences(device, stack.longs(inFlight)), "vkResetFences");

            IntBuffer pImageIndex = stack.mallocInt(1);
            check(vkAcquireNextImageKHR(device, swapchain, Long.MAX_VALUE, imageAvailable, VK_NULL_HANDLE,
                    pImageIndex), "vkAcquireNextImageKHR");
            int imageIndex = pImageIndex.get(0);

            check(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer");
            recordDraw(framebuffers[imageIndex], stack);

            VkSubmitInfo submit = VkSubmitInfo.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                    .waitSemaphoreCount(1)
                    .pWaitSemaphores(stack.longs(imageAvailable))
                    .pWaitDstStageMask(stack.ints(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT))
                    .pCommandBuffers(stack.pointers(commandBuffer))
                    .pSignalSemaphores(stack.longs(renderFinished));
            check(vkQueueSubmit(queue, submit, inFlight), "vkQueueSubmit");

            VkPresentInfoKHR present = VkPresentInfoKHR.calloc(stack)
                    .sType(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR)
                    .pWaitSemaphores(stack.longs(renderFinished))
                    .swapchainCount(1)
                    .pSwapchains(stack.longs(swapchain))
                    .pImageIndices(stack.ints(imageIndex));
            check(vkQueuePresentKHR(queue, present), "vkQueuePresentKHR");
        }
    }

    @Override
    public void close() {
        if (device != null) {
            vkDeviceWaitIdle(device);
            destroyIf(inFlight, h -> vkDestroyFence(device, h, null));
            destroyIf(renderFinished, h -> vkDestroySemaphore(device, h, null));
            destroyIf(imageAvailable, h -> vkDestroySemaphore(device, h, null));
            destroyIf(commandPool, h -> vkDestroyCommandPool(device, h, null));
            destroyIf(graphicsPipeline, h -> vkDestroyPipeline(device, h, null));
            destroyIf(pipelineLayout, h -> vkDestroyPipelineLayout(device, h, null));
            destroyIf(fragmentShaderModule, h -> vkDestroyShaderModule(device, h, null));
            destroyIf(vertexShaderModule, h -> vkDestroyShaderModule(device, h, null));
            for (long framebuffer : framebuffers) {
                vkDestroyFramebuffer(device, framebuffer, null);
            }
            destroyIf(renderPass, h -> vkDestroyRenderPass(device, h, null));
            for (long view : imageViews) {
                vkDestroyImageView(device, view, null);
            }
            destroyIf(swapchain, h -> vkDestroySwapchainKHR(device, h, null));
            vkDestroyDevice(device, null);
        }
        if (surface != VK_NULL_HANDLE && instance != null) {
            vkDestroySurfaceKHR(instance, surface, null);
        }
        if (instance != null) {
            vkDestroyInstance(instance, null);
        }
        if (window != NULL) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    }

    private interface Destroyer {
        void destroy(long handle);
    }

    private static void destroyIf(long handle, Destroyer destroyer) {
        if (handle != VK_NULL_HANDLE) {
            destroyer.destroy(handle);
        }
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void check(int result, String operation) {
        if (result != VK_SUCCESS) {
            throw new IllegalStateException(operation + " failed: VkResult " + result);
        }
    }
}
