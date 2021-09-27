#include "kernels.cuh"

torch::Tensor weighting_cuda_forward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_weight,
        const int kH, const int kW,
        const bool casual_mask
) {
    TypeCheck(x_ori);
    TypeCheck(x_weight);
    const int batch = x_weight.size(0);
    const int channels = x_ori.size(1);
    const int height = x_weight.size(1);
    const int width = x_weight.size(2);
    AT_ASSERTM(!casual_mask || (kH & 1 == 1 && kW & 1 == 1), "If casual_mask is true, the kernel size must be odd!");

    const int batch_ori = x_ori.size(0);
    const int channels_ori = x_ori.size(1);
    const int height_ori = x_ori.size(2);
    const int width_ori = x_ori.size(3);
    const int per_channel_ori = height_ori * width_ori;
    const int per_input_ori = per_channel_ori * channels_ori;
    AT_ASSERTM(batch % batch_ori == 0, "cannot use auto expand");
    AT_ASSERTM(height % height_ori == 0, "height cannot be divided exactly");
    AT_ASSERTM(width % width_ori == 0, "width cannot be divided exactly");
    const int ah = height / height_ori;
    const int aw = width / width_ori;

    const int rH = kH >> 1;
    const int rW = kW >> 1;
    const int patch = casual_mask ? (kH * kW >> 1) + 1: kH * kW;
    const int per_channel = height * width;
    const int per_input = per_channel * channels;
    const int per_output = per_channel * patch;
    auto output = torch::empty({batch, channels, height, width}, x_ori.options());

    // int start_inp = 0, start_out = 0, start_inp_ori = 0;
    // for (int j=0; j<batch_ori; ++j){
    //     for (int i=0; i<batch / batch_ori; ++i) {
    //         AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_ori.scalar_type(), "weighting_cuda_forward", 
    //             ([&] {
    //                 f_ck2c_ori<scalar_t, float> (
    //                         at::cuda::getCurrentCUDAStream(),
    //                         x_ori.data_ptr<scalar_t>() + start_inp_ori,
    //                         x_weight.data_ptr<scalar_t>() + start_out,
    //                         kH, kW, rH, rW,
    //                         patch, channels,
    //                         height_ori, width_ori,
    //                         per_channel_ori, per_input_ori,
    //                         output.data_ptr<scalar_t>() + start_inp,
    //                         ah, aw
    //                 );
    //             }
    //             )
    //             );
    //         start_inp += per_input;
    //         start_out += per_output;
    //     }
    //     start_inp_ori += per_input_ori;
    // }

    return output;
}

//////////////////////////////////////////////////////////////

torch::Tensor weighting_cuda_backward_ori(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_weight,
        const torch::Tensor &grad_out,
        const int kH, const int kW,
        const bool casual_mask
) {
    TypeCheck(x_weight);
    const int batch = x_weight.size(0);
    const int channels = grad_out.size(1);
    const int height = x_weight.size(1);
    const int width = x_weight.size(2);

    const int batch_ori = x_ori.size(0);
    const int channels_ori = x_ori.size(1);
    const int height_ori = x_ori.size(2);
    const int width_ori = x_ori.size(3);
    const int per_channel_ori = height_ori * width_ori;
    const int per_input_ori = per_channel_ori * channels_ori;
    AT_ASSERTM(batch % batch_ori == 0, "cannot use auto expand");
    AT_ASSERTM(height % height_ori == 0, "height cannot be divided exactly");
    AT_ASSERTM(width % width_ori == 0, "width cannot be divided exactly");
    const int ah = height / height_ori;
    const int aw = width / width_ori;

    const int rH = kH >> 1;
    const int rW = kW >> 1;
    const int patch = casual_mask ? (kH * kW >> 1) + 1: kH * kW;
    const int per_channel = height * width;
    const int per_input = per_channel * channels;
    const int per_output = per_channel * patch;
    auto grad_ori = torch::empty({batch_ori, channels, height_ori, width_ori}, x_weight.options());

    int start_inp = 0, start_out = 0;
    for (int j=0; j<batch_ori; ++j){
        bool is_accumulate = false;
        for (int i=0; i<batch / batch_ori; ++i) {
            auto grad_out_row = grad_out.select(0, i + j * batch / batch_ori);
            
            // AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_weight.scalar_type(), "weighting_cuda_backward_ori", 
            //     ([&] {
            //         f_ck2c_loc<scalar_t, float> (
            //                 at::cuda::getCurrentCUDAStream(),
            //                 grad_out_row.data_ptr<scalar_t>(),
            //                 x_weight.data_ptr<scalar_t>() + start_out,
            //                 kH, kW, rH, rW,
            //                 patch, channels,
            //                 height_ori, width_ori,
            //                 per_channel_ori, per_input_ori,
            //                 grad_ori.data_ptr<scalar_t>() + start_inp,
            //                 is_accumulate,
            //                 ah, aw
            //         );
            //     }
            //     )
            //     );
            start_out += per_output;
            is_accumulate = true;
        }
        start_inp += per_input_ori;
    }
    return grad_ori;
}

//////////////////////////////////////////////////////////////

torch::Tensor weighting_cuda_backward_weight(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_weight,
        const torch::Tensor &grad_out,
        const int kH, const int kW,
        const bool casual_mask
) {
    TypeCheck(x_ori);
    const int batch = grad_out.size(0);
    const int channels = x_ori.size(1);
    const int height = grad_out.size(2);
    const int width = grad_out.size(3);

    const int batch_ori = x_ori.size(0);
    const int channels_ori = x_ori.size(1);
    const int height_ori = x_ori.size(2);
    const int width_ori = x_ori.size(3);
    const int per_channel_ori = height_ori * width_ori;
    const int per_input_ori = per_channel_ori * channels_ori;
    AT_ASSERTM(batch % batch_ori == 0, "cannot use auto expand");
    AT_ASSERTM(height % height_ori == 0, "height cannot be divided exactly");
    AT_ASSERTM(width % width_ori == 0, "width cannot be divided exactly");
    const int ah = height / height_ori;
    const int aw = width / width_ori;

    const int rH = kH >> 1;
    const int rW = kW >> 1;
    const int patch = casual_mask ? (kH * kW >> 1) + 1: kH * kW;
    const int per_channel = height * width;
    const int per_input = per_channel * channels;
    const int per_output = per_channel * patch;
    auto grad_weight = torch::empty({batch, height, width, patch}, x_ori.options());

    int start_inp = 0, start_out = 0;
    // for (int j=0; j<batch_ori; j++){

    //     for (int i=0; i<batch / batch_ori; ++i) {
    //         auto grad_out_row = grad_out.select(0, i + j * batch / batch_ori);

    //         AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_ori.scalar_type(), "weighting_cuda_backward_weight", 
    //             ([&] {
    //                 f_cc2k<scalar_t, float> (
    //                         at::cuda::getCurrentCUDAStream(),
    //                         grad_out_row.data_ptr<scalar_t>(),
    //                         x_ori.data_ptr<scalar_t>() + start_inp,
    //                         kH, kW, rH, rW,
    //                         patch, channels,
    //                         height_ori, width_ori,
    //                         per_channel_ori,
    //                         grad_weight.data_ptr<scalar_t>() + start_out,
    //                         ah, aw
    //                 );
    //             }
    //             )
    //             );
    //         start_out += per_output;
    //     }
    //     start_inp += per_input_ori;
    // }

    return grad_weight;
}