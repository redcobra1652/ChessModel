// mcts.cpp
//
// Minimal, self-contained, single-threaded AlphaZero-style MCTS engine.
//
// This process is a long-lived "MCTS server". It is started once by
// train.py and reused for every self-play / evaluation move for the
// entire run. It knows NOTHING about chess rules itself -- python-chess
// (driven from train.py) is the single source of truth for legality,
// move application, and terminal detection. This engine only manages
// the search tree (selection / PUCT / virtual loss / backup) and asks
// train.py, over a line-based JSON protocol on stdin/stdout, whenever
// it needs to know the legal moves + policy priors + value + terminal
// status of a position, or the resulting FEN of playing a move.
//
// PROTOCOL (one JSON object per line, always flushed):
//
//   train.py -> engine (commands):
//     {"cmd":"search","fen":"<fen>","sims":800,"threads":4}
//     {"cmd":"quit"}
//
//   engine -> train.py (requests, sent while a search is running):
//     {"type":"root","fen":"<fen>"}
//         -> response: {"terminal":bool,"result":num,
//                        "moves":[uci,...],"priors":[num,...],"value":num}
//     {"type":"visit","fen":"<parent_fen>","move":"<uci>"}
//         -> response: {"fen":"<child_fen>","terminal":bool,"result":num,
//                        "moves":[uci,...],"priors":[num,...],"value":num}
//
//   engine -> train.py (final answer to a "search" command):
//     {"cmd":"result","moves":[uci,...],"visits":[num,...],"value":num}
//
// Build (macOS):
//   clang++ -O3 -std=c++17 -pthread -o mcts_engine mcts.cpp
//
// This file has ZERO external dependencies -- only the C++ standard
// library -- including a small hand-rolled JSON reader/writer, since the
// message schema above is fixed and simple.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// ----------------------------------------------------------------------
// Tiny JSON value + single-line object parser/writer
// ----------------------------------------------------------------------

struct JsonVal {
    enum Type { STR, NUM, BOOL, ARR_STR, ARR_NUM, NUL } type = NUL;
    std::string s;
    double n = 0.0;
    bool b = false;
    std::vector<std::string> arr_s;
    std::vector<double> arr_n;

    static JsonVal Str(const std::string& v) { JsonVal j; j.type = STR; j.s = v; return j; }
    static JsonVal Num(double v) { JsonVal j; j.type = NUM; j.n = v; return j; }
    static JsonVal Bool(bool v) { JsonVal j; j.type = BOOL; j.b = v; return j; }
    static JsonVal ArrStr(const std::vector<std::string>& v) { JsonVal j; j.type = ARR_STR; j.arr_s = v; return j; }
    static JsonVal ArrNum(const std::vector<double>& v) { JsonVal j; j.type = ARR_NUM; j.arr_n = v; return j; }

    std::string get_str(const std::string& def = "") const { return type == STR ? s : def; }
    double get_num(double def = 0.0) const { return type == NUM ? n : def; }
    bool get_bool(bool def = false) const { return type == BOOL ? b : def; }
};

using JsonObj = std::map<std::string, JsonVal>;

// ---- Writer ----

static std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out;
}

static std::string json_write(const JsonObj& obj) {
    std::ostringstream oss;
    oss << '{';
    bool first = true;
    for (const auto& kv : obj) {
        if (!first) oss << ',';
        first = false;
        oss << '"' << kv.first << "\":";
        const JsonVal& v = kv.second;
        switch (v.type) {
            case JsonVal::STR:
                oss << '"' << json_escape(v.s) << '"';
                break;
            case JsonVal::NUM:
                oss << v.n;
                break;
            case JsonVal::BOOL:
                oss << (v.b ? "true" : "false");
                break;
            case JsonVal::ARR_STR: {
                oss << '[';
                for (size_t i = 0; i < v.arr_s.size(); ++i) {
                    if (i) oss << ',';
                    oss << '"' << json_escape(v.arr_s[i]) << '"';
                }
                oss << ']';
                break;
            }
            case JsonVal::ARR_NUM: {
                oss << '[';
                for (size_t i = 0; i < v.arr_n.size(); ++i) {
                    if (i) oss << ',';
                    oss << v.arr_n[i];
                }
                oss << ']';
                break;
            }
            case JsonVal::NUL:
                oss << "null";
                break;
        }
    }
    oss << '}';
    return oss.str();
}

// ---- Parser (enough for our fixed schema: flat object of
//      string / number / bool / array-of-string / array-of-number) ----

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : s_(text), i_(0), n_(text.size()) {}

    JsonObj parse_object() {
        JsonObj obj;
        skip_ws();
        expect('{');
        skip_ws();
        if (peek() == '}') { ++i_; return obj; }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            skip_ws();
            JsonVal val = parse_value();
            obj[key] = val;
            skip_ws();
            char c = peek();
            if (c == ',') { ++i_; continue; }
            if (c == '}') { ++i_; break; }
            break; // malformed; stop gracefully
        }
        return obj;
    }

private:
    const std::string& s_;
    size_t i_;
    size_t n_;

    char peek() const { return i_ < n_ ? s_[i_] : '\0'; }

    void expect(char c) {
        if (peek() == c) ++i_;
    }

    void skip_ws() {
        while (i_ < n_ && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
    }

    std::string parse_string() {
        std::string out;
        if (peek() != '"') return out;
        ++i_; // opening quote
        while (i_ < n_ && s_[i_] != '"') {
            char c = s_[i_];
            if (c == '\\' && i_ + 1 < n_) {
                char nx = s_[i_ + 1];
                switch (nx) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    default: out += nx;
                }
                i_ += 2;
            } else {
                out += c;
                ++i_;
            }
        }
        if (i_ < n_) ++i_; // closing quote
        return out;
    }

    double parse_number() {
        size_t start = i_;
        if (peek() == '-' || peek() == '+') ++i_;
        while (i_ < n_ && (isdigit((unsigned char)s_[i_]) || s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-')) ++i_;
        std::string tok = s_.substr(start, i_ - start);
        try { return std::stod(tok); } catch (...) { return 0.0; }
    }

    JsonVal parse_value() {
        skip_ws();
        char c = peek();
        if (c == '"') {
            return JsonVal::Str(parse_string());
        } else if (c == '[') {
            return parse_array();
        } else if (c == 't') {
            i_ += 4; // true
            return JsonVal::Bool(true);
        } else if (c == 'f') {
            i_ += 5; // false
            return JsonVal::Bool(false);
        } else if (c == 'n') {
            i_ += 4; // null
            return JsonVal();
        } else {
            return JsonVal::Num(parse_number());
        }
    }

    JsonVal parse_array() {
        std::vector<std::string> arr_s;
        std::vector<double> arr_n;
        bool is_str_arr = false;
        bool any = false;
        ++i_; // '['
        skip_ws();
        if (peek() == ']') { ++i_; return JsonVal::ArrNum(arr_n); }
        while (true) {
            skip_ws();
            if (peek() == '"') {
                is_str_arr = true;
                arr_s.push_back(parse_string());
            } else {
                arr_n.push_back(parse_number());
            }
            any = true;
            skip_ws();
            char c = peek();
            if (c == ',') { ++i_; continue; }
            if (c == ']') { ++i_; break; }
            break;
        }
        (void)any;
        if (is_str_arr) return JsonVal::ArrStr(arr_s);
        return JsonVal::ArrNum(arr_n);
    }
};

static JsonObj json_parse(const std::string& line) {
    JsonParser p(line);
    return p.parse_object();
}

// ----------------------------------------------------------------------
// Serialized stdin/stdout communication with train.py
//
// Only ONE thread may be writing a request and awaiting its response
// on the pipe at any given time. This mutex enforces strict
// alternation (write request -> block on read response) which is what
// keeps the protocol deadlock-free even though multiple MCTS worker
// threads run concurrently.
// ----------------------------------------------------------------------

static std::mutex g_io_mutex;

static JsonObj send_request(const JsonObj& req) {
    std::lock_guard<std::mutex> lock(g_io_mutex);
    std::cout << json_write(req) << std::endl; // std::endl flushes
    std::string line;
    if (!std::getline(std::cin, line)) {
        // Peer closed the pipe unexpectedly.
        std::cerr << "[mcts_engine] FATAL: stdin closed while awaiting response" << std::endl;
        std::exit(1);
    }
    return json_parse(line);
}

// ----------------------------------------------------------------------
// MCTS tree
// ----------------------------------------------------------------------

constexpr double C_PUCT = 1.5;
constexpr int VIRTUAL_LOSS = 3;

// Mate-distance shaping: terminal values are scaled down slightly per
// ply of depth below the root, so that a forced mate in 2 plies scores
// strictly higher (for the winner) than an otherwise-equal forced mate
// in 20 plies, instead of both saturating to the same +1/-1 and tying
// in the backed-up Q value. 0.003/ply keeps the effect small enough
// that it never overrides a genuine difference in game-theoretic
// outcome (win/loss/draw) -- it only breaks ties *within* same-outcome
// lines by preferring the faster one. Clamped in expand_node() so it
// can never flip the sign of the result even at very high depth.
constexpr double MATE_DECAY_K = 0.003;

struct Node {
    Node* parent = nullptr;
    std::string move_uci;           // move that led from parent to this node ("" for root)
    std::string fen;                // known for root immediately; for others, filled on first expansion
    // Real game move history (UCI, from game start) leading up to this
    // search's root position. Only ever populated on the root node itself
    // and read off `root->history` elsewhere -- kept on Node rather than
    // as a bare global so a future multi-search-in-flight design isn't
    // blocked by it. Threaded into every root/visit request (together
    // with each node's own tree path, see build_path()) so train.py can
    // replay real history and correctly detect threefold repetition --
    // something a bare FEN can never represent.
    std::vector<std::string> history;
    std::map<std::string, std::unique_ptr<Node>> children;

    int N = 0;
    double W = 0.0;
    double P = 0.0;

    // Ply-depth of this node below the search root (root itself is 0).
    // Used only for mate-distance shaping of terminal values -- see
    // MATE_DECAY_K and its use in expand_node().
    int depth = 0;

    bool expanded = false;
    bool terminal = false;
    double terminal_value = 0.0;    // value from this node's side-to-move perspective

    std::mutex mtx;
};

// Walks from `node` up to (but not including) the root, collecting the
// move that led to each node along the way, and returns them in
// root-to-node order. Combined with root->history, this fully identifies
// the hypothetical line reached by this search path -- needed so
// train.py can replay real moves and detect repetition correctly for
// positions that only exist inside the search tree, not just in the
// actual game played so far.
static std::vector<std::string> build_path(Node* node) {
    std::vector<std::string> path;
    for (Node* n = node; n->parent != nullptr; n = n->parent) {
        path.push_back(n->move_uci);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

// Expand `node` (assumes node->mtx is NOT held). Populates children /
// terminal status / value by querying train.py exactly once.
// Returns the leaf value from `node`'s own side-to-move perspective.
static double expand_node(Node* node, Node* root) {
    JsonObj resp;
    if (node == root) {
        JsonObj req;
        req["type"] = JsonVal::Str("root");
        req["history"] = JsonVal::ArrStr(root->history);
        resp = send_request(req);
    } else {
        JsonObj req;
        req["type"] = JsonVal::Str("visit");
        req["history"] = JsonVal::ArrStr(root->history);
        req["path"] = JsonVal::ArrStr(build_path(node->parent));
        req["move"] = JsonVal::Str(node->move_uci);
        resp = send_request(req);
    }

    std::lock_guard<std::mutex> lock(node->mtx);
    if (node->expanded) {
        // Another thread already expanded this node while we were
        // waiting on the response (double-checked lock). Use its result.
        return node->terminal ? node->terminal_value : 0.0; // caller re-reads under lock anyway
    }

    if (node != root) {
        node->fen = resp.count("fen") ? resp["fen"].get_str() : node->fen;
    }
    node->terminal = resp.count("terminal") ? resp["terminal"].get_bool() : false;

    double value;
    if (node->terminal) {
        double raw_result = resp.count("result") ? resp["result"].get_num() : 0.0;
        if (raw_result != 0.0) {
            // Shrink magnitude slightly with depth (see MATE_DECAY_K),
            // clamped so the sign/outcome itself is never altered.
            double shaped = raw_result * (1.0 - MATE_DECAY_K * node->depth);
            shaped = std::max(-1.0, std::min(1.0, shaped));
            // Guard against decay ever crossing zero for pathologically
            // deep terminal nodes -- keep at least a sliver of the
            // original sign/magnitude.
            if ((raw_result > 0.0 && shaped <= 0.0) || (raw_result < 0.0 && shaped >= 0.0)) {
                shaped = raw_result * 0.01;
            }
            node->terminal_value = shaped;
        } else {
            node->terminal_value = 0.0;
        }
        value = node->terminal_value;
    } else {
        const std::vector<std::string>& moves = resp["moves"].arr_s;
        const std::vector<double>& priors = resp["priors"].arr_n;
        for (size_t i = 0; i < moves.size(); ++i) {
            auto child = std::make_unique<Node>();
            child->parent = node;
            child->move_uci = moves[i];
            child->P = (i < priors.size()) ? priors[i] : 0.0;
            child->depth = node->depth + 1;
            node->children[moves[i]] = std::move(child);
        }
        value = resp.count("value") ? resp["value"].get_num() : 0.0;
    }
    node->expanded = true;
    return value;
}

// Runs a single MCTS simulation from `root`.
static void simulate_one(Node* root) {
    std::vector<Node*> path;
    Node* node = root;
    path.push_back(node);
    double leaf_value = 0.0;

    while (true) {
        std::unique_lock<std::mutex> lock(node->mtx);

        if (node->terminal) {
            leaf_value = node->terminal_value;
            lock.unlock();
            break;
        }

        if (!node->expanded) {
            lock.unlock();
            leaf_value = expand_node(node, root);
            // Re-read under lock in case of a race on terminal_value.
            std::lock_guard<std::mutex> lk2(node->mtx);
            if (node->terminal) leaf_value = node->terminal_value;
            break;
        }

        if (node->children.empty()) {
            // Expanded but has no children: treat conservatively as a
            // drawn/neutral leaf (should not normally happen since the
            // terminal flag is expected to catch these cases upstream).
            leaf_value = 0.0;
            lock.unlock();
            break;
        }

        double sqrt_parent_n = std::sqrt(static_cast<double>(std::max(1, node->N)));
        double best_score = -1e18;
        Node* best_child = nullptr;
        for (auto& kv : node->children) {
            Node* c = kv.second.get();
            std::lock_guard<std::mutex> clk(c->mtx);
            double q = (c->N > 0) ? -(c->W / c->N) : 0.0;
            double u = C_PUCT * c->P * sqrt_parent_n / (1.0 + c->N);
            double score = q + u;
            if (score > best_score) {
                best_score = score;
                best_child = c;
            }
        }

        {
            std::lock_guard<std::mutex> clk(best_child->mtx);
            best_child->N += VIRTUAL_LOSS;
            best_child->W += VIRTUAL_LOSS;
        }

        lock.unlock();
        node = best_child;
        path.push_back(node);
    }

    // Backup: alternate sign since each depth level flips side-to-move.
    double v = leaf_value;
    for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
        Node* n = path[i];
        std::lock_guard<std::mutex> lk(n->mtx);
        if (i != 0) {
            // undo the virtual loss this node received when it was
            // selected as a child during the descent above
            n->N -= VIRTUAL_LOSS;
            n->W -= VIRTUAL_LOSS;
        }
        n->N += 1;
        n->W += v;
        v = -v;
    }
}

// ----------------------------------------------------------------------
// main
// ----------------------------------------------------------------------

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    std::ios::sync_with_stdio(true);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        JsonObj cmd = json_parse(line);
        std::string c = cmd.count("cmd") ? cmd["cmd"].get_str() : "";

        if (c == "quit") {
            break;
        } else if (c == "search") {
            std::string fen = cmd["fen"].get_str();
            int sims = static_cast<int>(cmd.count("sims") ? cmd["sims"].get_num() : 200);
            int nthreads = static_cast<int>(cmd.count("threads") ? cmd["threads"].get_num() : 4);
            nthreads = std::max(1, nthreads);
            sims = std::max(1, sims);

            auto root = std::make_unique<Node>();
            root->fen = fen;
            root->history = cmd.count("history") ? cmd["history"].arr_s : std::vector<std::string>{};

            // All simulations run on THIS thread (the main thread that owns
            // stdin/stdout). Running them in a thread pool was a deadlock:
            // each sim calls send_request() which does blocking I/O, but
            // main() was also blocked on std::getline -- so the worker
            // threads' requests were landing in main()'s getline and being
            // treated as unknown top-level commands. Serial execution here
            // is correct; MCTS is inherently sequential through the pipe.
            for (int i = 0; i < sims; ++i) {
                simulate_one(root.get());
            }

            std::vector<std::string> moves;
            std::vector<double> visits;
            double root_value = 0.0;
            {
                std::lock_guard<std::mutex> lk(root->mtx);
                if (root->N > 0) root_value = root->W / root->N;
                for (auto& kv : root->children) {
                    moves.push_back(kv.first);
                    std::lock_guard<std::mutex> clk(kv.second->mtx);
                    visits.push_back(static_cast<double>(kv.second->N));
                }
            }

            JsonObj result;
            result["cmd"] = JsonVal::Str("result");
            result["moves"] = JsonVal::ArrStr(moves);
            result["visits"] = JsonVal::ArrNum(visits);
            result["value"] = JsonVal::Num(root_value);

            std::lock_guard<std::mutex> iolock(g_io_mutex);
            std::cout << json_write(result) << std::endl;
        } else {
            std::cerr << "[mcts_engine] WARNING: unknown command: " << line << std::endl;
        }
    }

    return 0;
}
