#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

struct Options
{
    bool show_help{false};
    std::string hash_hex;
    std::string charset{"lower"};
    int min_len{1};
    int max_len{1};
    std::string ui_mode{"none"};
};

Options parse_args(int argc, char **argv)
{
    Options opts;
    std::unordered_map<std::string, std::string> kv;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto consume_value = [&](const std::string &key) -> std::string
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("Missing value for " + key);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--help" || arg == "-h")
        {
            opts.show_help = true;
        }
        else if (arg == "--hash")
        {
            opts.hash_hex = consume_value(arg);
        }
        else if (arg == "--charset")
        {
            opts.charset = consume_value(arg);
        }
        else if (arg == "--min-len")
        {
            opts.min_len = std::stoi(consume_value(arg));
        }
        else if (arg == "--max-len")
        {
            opts.max_len = std::stoi(consume_value(arg));
        }
        else if (arg == "--ui")
        {
            opts.ui_mode = consume_value(arg);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return opts;
}

void print_ascii_banner()
{
    std::cout << R"(
                    _.--._
               _.-.'      `.-._
             .' ./`--...--' \  `.
    .-.      `.'.`--.._..--'   .'
_..'.-.`-._.'( (-..__    __..-'
 >.'   `-...' ) )    ````
 '           / /     "brute force, but with a hat"
        .._.'.'    
         >.-'   v0.1.0 - Åbo Akademi University, 2025
         '
  /\  /\__ _ ___| |__ | |__   __ _| |_ 
 / /_/ / _` / __| '_ \| '_ \ / _` | __|
/ __  / (_| \__ \ | | | | | | (_| | |_ 
\/ /_/ \__,_|___/_| |_|_| |_|\__,_|\__|
)" << "\n";
}

void print_help()
{
    print_ascii_banner();
    std::cout << "Hashhat - CUDA GPU Password Cracker (CLI scaffold)\n\n";
    std::cout << "Usage: hashhat [--hash <hex>] [--charset list] [--min-len N] [--max-len N] [--ui curses|none]\n";
    std::cout << "\n";
    std::cout << "Flags (current placeholders):\n";
    std::cout << "  --hash <hex>        Target hash in hex (MD5 initially)\n";
    std::cout << "  --charset list      Comma list: lower,upper,num,sym (default: lower)\n";
    std::cout << "  --min-len N         Minimum password length (default: 1)\n";
    std::cout << "  --max-len N         Maximum password length (default: 1)\n";
    std::cout << "  --ui <mode>         Optional UI mode: none|curses (default: none)\n";
    std::cout << "  --help              Show this help\n";
    std::cout << "\n";
    std::cout << "Roadmap hooks:\n";
    std::cout << "  - TODO: wire GPU kernel launch here\n";
    std::cout << "  - TODO: add CPU baseline and correctness checks\n";
    std::cout << "  - TODO: integrate curses UI when --ui=curses\n";
}

void print_options(const Options &o)
{
    std::cout << "Hashhat options:\n";
    std::cout << "  hash:     " << (o.hash_hex.empty() ? "(none)" : o.hash_hex) << "\n";
    std::cout << "  charset:  " << o.charset << "\n";
    std::cout << "  min-len:  " << o.min_len << "\n";
    std::cout << "  max-len:  " << o.max_len << "\n";
    std::cout << "  ui:       " << o.ui_mode << "\n";
}

int main(int argc, char **argv)
{
    try
    {
        Options opts = parse_args(argc, argv);

        if (opts.show_help)
        {
            print_help();
            return 0;
        }

        if (opts.min_len > opts.max_len)
        {
            throw std::runtime_error("min-len cannot exceed max-len");
        }

        print_options(opts);

        // TODO 1: Wire GPU kernel launch (hash generation + compare) using options above.
        // TODO 2: Add CPU baseline for correctness and simple benchmark output.
        // TODO 3: Integrate curses dashboard when opts.ui_mode == "curses".
        // TODO 4: Add configurable hash algorithm selection (MD5 now, SHA variants later).
        // TODO 5: Move charset/hash data to constant memory and measure speedups.
        // TODO 6: Add test hooks for small keyspaces to validate end-to-end.

        std::cout << "\n[stub] No cracking implemented yet. See TODOs in source.\n";
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        std::cerr << "Use --help for usage.\n";
        return 1;
    }
}
