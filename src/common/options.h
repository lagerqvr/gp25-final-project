#pragma once

#include <string>

struct Options
{
    bool show_help{false};
    bool run_benchmark{false};
    std::string hash_hex;
    std::string algo{"sha1"};
    std::string charset{"lower"};
    int min_len{1};
    int max_len{1};
    std::string ui_mode{"none"};
    std::string device_name;
    int block_size{256};
    std::string target_hex;
};
