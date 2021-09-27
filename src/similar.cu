#include "kernels.cuh"
using namespace at;

torch::Tensor similar_cuda_forward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_loc,
        const int kH, const int kW,
        const bool casual_mask
) {
    TypeCheck(x_ori);
    TypeCheck(x_loc);
    AT_ASSERTM(!casual_mask || (kH & 1 == 1 && kW & 1 == 1), "If casual_mask is true, the kernel size must be odd!");
    const int batch = x_ori.size(0);
    const int channels = x_ori.size(1);
    const int height = x_ori.size(2);
    const int width = x_ori.size(3);

    const int batch_loc = x_loc.size(0);
    const int channels_loc = x_loc.size(1);
    const int height_loc = x_loc.size(2);
    const int width_loc = x_loc.size(3);
    const int per_channel_loc = height_loc * width_loc;
    const int per_input_loc = per_channel_loc * channels_loc;
    AT_ASSERTM(batch == batch_loc, "batch size should be same.");
    AT_ASSERTM(height % height_loc == 0, "height cannot be divided exactly");
    AT_ASSERTM(width % width_loc == 0, "width cannot be divided exactly");
    const int ah = height / height_loc;
    const int aw = width / width_loc;

    const int rH = kH >> 1;
    const int rW = kW >> 1;
    const int patch = casual_mask ? (kH * kW >> 1) + 1: kH * kW;
    const int per_channel = height * width;
    const int per_input = per_channel * channels;
    const int per_output = height * width * patch;
    auto output = torch::empty({batch, height, width, patch}, x_ori.options());

            AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_ori.scalar_type(), "similar_cuda_forward", 
                ([&] {
                        f_cc2k<scalar_t, float>(
                            at::cuda::getCurrentCUDAStream(),
                            x_ori.data_ptr<scalar_t>(),
                            x_loc.data_ptr<scalar_t>(),
                            kH, kW, rH, rW,
                            patch, channels, height_loc, width_loc,
                            per_channel_loc, batch,
                            output.data_ptr<scalar_t>(), ah, aw, true
                        );
                }
                )
            );

    return output;
}

//////////////////////////////////////////////////////////////

torch::Tensor similar_cuda_backward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_loc,
        const torch::Tensor &grad_out,
        const int kH, const int kW,
        const bool is_ori,
        const bool casual_mask
) {
    TypeCheck(x_ori);
    TypeCheck(x_loc);
    const int batch = x_ori.size(0);
    const int channels = x_ori.size(1);
    const int height = x_ori.size(2);
    const int width = x_ori.size(3);

    const int rH = kH >> 1;
    const int rW = kW >> 1;
    const int patch = casual_mask ? (kH * kW >> 1) + 1: kH * kW;
    const int per_channel = height * width;
    const int per_input = per_channel * channels;

    const int batch_loc = x_loc.size(0);
    const int channels_loc = x_loc.size(1);
    const int height_loc = x_loc.size(2);
    const int width_loc = x_loc.size(3);
    const int per_channel_loc = height_loc * width_loc;
    const int per_input_loc = per_channel_loc * channels_loc;
    AT_ASSERTM(batch == batch_loc, "cannot use auto expand");
    AT_ASSERTM(height % height_loc == 0, "height cannot be divided exactly");
    AT_ASSERTM(width % width_loc == 0, "width cannot be divided exactly");
    const int ah = height / height_loc;
    const int aw = width / width_loc;

    if (is_ori){ // x is loc
        const torch::Tensor & x = x_loc;
        auto grad_inp = torch::empty({batch, channels, height, width}, x.options());
        int start_inp = 0, start_inp_loc = 0;
        for (int j = 0; j < batch_loc; ++j) {
            for (int i = 0; i < batch / batch_loc; ++i) {
                auto grad_out_row = grad_out.select(0, i + j * batch / batch_loc);
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "similar_cuda_backward_ori", 
                    ([&] {
                            f_ck2c_ori<scalar_t, float>(
                                at::cuda::getCurrentCUDAStream(),
                                x.data_ptr<scalar_t>() + start_inp_loc,
                                grad_out_row.data_ptr<scalar_t>(),
                                kH, kW, rH, rW,
                                patch, channels,
                                height_loc, width_loc,
                                per_channel_loc, per_input_loc,
                                grad_inp.data_ptr<scalar_t>() + start_inp, ah, aw
                            );
                    }
                    )
                    );
                start_inp += per_input;
            }
            start_inp_loc += per_input_loc;
        }
        return grad_inp;
    } else{ // x is ori
        const torch::Tensor & x = x_ori;
        auto grad_inp = torch::empty({batch_loc, channels_loc, height_loc, width_loc}, x.options());
        
                AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "similar_cuda_backward_loc", 
                ([&] {
                        f_ck2c_loc<scalar_t, float>(
                                at::cuda::getCurrentCUDAStream(),
                                x.data_ptr<scalar_t>(),
                                grad_out.data_ptr<scalar_t>(),
                                kH, kW, rH, rW,
                                patch, channels,
                                height_loc, width_loc,
                                per_channel_loc, per_input_loc, batch,
                                grad_inp.data_ptr<scalar_t>(),
                                false, ah, aw
                        );
                }
                )
                );
                
        return grad_inp;
    }
    
}
