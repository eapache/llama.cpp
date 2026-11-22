#include "llama-ext.h"
#include "ggml-backend.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

static void memory(llama_context * ctx, int pos) {
    size_t kv = 0;
    for (auto buffer : llama_memory_get_buffers(llama_get_memory(ctx))) {
        if (std::string(ggml_backend_buffer_name(buffer)) == "CUDA0") kv += ggml_backend_buffer_get_size(buffer);
    }
    for (const auto & [type, m] : llama_get_memory_breakdown(ctx)) {
        if (std::string(ggml_backend_buft_name(type)) == "CUDA0") {
            std::cout << pos << " " << m.model << " " << m.context << " " << m.compute << " " << kv << "\n";
        }
    }
}

int main(int argc, char ** argv) {
    if (argc != 7 && argc != 8) return 1;
    ggml_backend_load_all_from_path(argv[1]);
    llama_backend_init();
    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name("CUDA0"), nullptr};
    if (!devices[0]) throw std::runtime_error("CUDA0 missing");
    mp.devices = devices;
    mp.n_gpu_layers = 999;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    auto model = llama_model_load_from_file(argv[2], mp);
    if (!model) return 2;
    const auto vocab = llama_model_get_vocab(model);
    std::ifstream input(argv[5]);
    std::string text(std::istreambuf_iterator<char>(input), {});
    std::vector<llama_token> tokens(text.size()+16);
    int n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, true);
    if (n < 2560) throw std::runtime_error("trace too short");
    auto cp = llama_context_default_params();
    cp.n_ctx = std::stoi(argv[3]);
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.kv_unified = true;
    cp.swa_full = false;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    std::string mode = argv[4];
    cp.type_k = cp.type_v = mode == "q8" ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
    if (mode == "lazy") {
        cp.kv_lazy = true;
        cp.type_k_initial = cp.type_v_initial = GGML_TYPE_Q8_0;
    }
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) return 3;
    memory(ctx, 0);
    {
        std::ofstream output(argv[6], std::ios::binary);
        auto batch = llama_batch_ext_init(ctx);
        for (int pos = 0; pos < (cp.n_ctx == 3072 ? 2560 : 128);  ) {
            const int chunk = pos == 0 && argc == 8 ? std::stoi(argv[7]) : 128;
            llama_batch_ext_clear(batch);
            for (int j = 0; j < chunk; ++j) {
                int row = llama_batch_ext_add_token(batch, 0, tokens[pos+j]);
                llama_pos position = pos+j;
                if (row < 0 || !llama_batch_ext_set_pos(batch, row, &position)) return 4;
                if (!llama_batch_ext_set_output_logits(batch, row, j == chunk-1)) return 4;
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) return 5;
            auto logits = llama_get_logits_ith(ctx, -1);
            if (!logits) return 6;
            output.write(reinterpret_cast<const char *>(logits), llama_vocab_n_tokens(vocab)*sizeof(float));
            pos += chunk;
            memory(ctx, pos);
        }
        if (!output) return 7;
        llama_batch_ext_free(batch);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
