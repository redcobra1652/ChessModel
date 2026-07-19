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
//
// Batching extension (backward-compatible):
//   C++ -> Python:  {"cmd":"batch","requests":[{type,history,path,move},...]}
//   Python -> C++:  {"cmd":"batch_result","responses":["{...}","..."]}
//
// The root is still evaluated with a single send_request() call because
// it must complete before any simulations start and batching it saves
// nothing.  All non-root leaf evals are batched BATCH_SIZE at a time.
// ----------------------------------------------------------------------

static std::mutex g_io_mutex;

// Single-request round-trip. Used for the root call only.
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

static constexpr int BATCH_SIZE = 8;

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
    bool in_flight = false;          // true while queued in a pending batch (not yet expanded)
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

// Apply a Python NN response to a node: populate children / terminal /
// value.  Returns the leaf value from the node's side-to-move perspective.
// Mirrors the logic that was in expand_node() in the original.
static double apply_response(Node* node, Node* root, JsonObj resp) {
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
    node->in_flight = false;
    return value;
}

// Holds the state of one in-progress simulation that has reached an
// unexpanded leaf and is waiting for its NN evaluation.
struct PendingSim {
    std::vector<Node*> path;    // traversal path from root to leaf (inclusive)
    Node*              leaf;    // the unexpanded leaf node
    JsonObj            request; // the visit request to send to Python
};

// Descend the tree using PUCT until we hit an unexpanded leaf or terminal.
//
// Returns true  if the sim resolved inline (terminal, or expanded-but-empty
//               leaf treated as draw): path has been backed up already.
// Returns false if we landed on an unexpanded leaf that needs an NN eval:
//               `pending` is populated; caller must send the request and
//               call backup() once the response arrives.
//
// BUG FIX vs previous batching attempt: when a second descent lands on an
// in_flight node we no longer back up a spurious 0.0 and count it as
// completed. Instead we return a special sentinel (return value = false,
// pending.leaf = nullptr) so run_sims() can simply retry that descent slot
// without corrupting N/W anywhere in the tree.
static bool descend(Node* root, PendingSim& pending) {
    std::vector<Node*> path;
    Node* node = root;
    path.push_back(node);

    while (true) {
        // Terminal: back up immediately, no NN eval needed.
        if (node->terminal) {
            double v = node->terminal_value;
            for (int i = (int)path.size() - 1; i >= 0; --i) {
                Node* n = path[i];
                if (i != 0) { n->N -= VIRTUAL_LOSS; n->W -= VIRTUAL_LOSS; }
                n->N += 1; n->W += v; v = -v;
            }
            return true;
        }

        // Unexpanded leaf: needs NN eval. Queue for batch.
        // (in_flight cannot happen here: PUCT below skips in_flight children,
        // so no two descents in the same batch window reach the same node.)
        if (!node->expanded) {
            JsonObj req;
            req["type"] = JsonVal::Str("visit");
            req["history"] = JsonVal::ArrStr(root->history);
            req["path"] = JsonVal::ArrStr(build_path(node->parent));
            req["move"] = JsonVal::Str(node->move_uci);
            node->in_flight = true;
            pending.path = std::move(path);
            pending.leaf = node;
            pending.request = std::move(req);
            return false;
        }

        // Expanded but no children (stalemate/draw caught late): draw.
        if (node->children.empty()) {
            double v = 0.0;
            for (int i = (int)path.size() - 1; i >= 0; --i) {
                Node* n = path[i];
                if (i != 0) { n->N -= VIRTUAL_LOSS; n->W -= VIRTUAL_LOSS; }
                n->N += 1; n->W += v; v = -v;
            }
            return true;
        }

        // PUCT selection.
        // Skip in_flight children: they are unexpanded nodes already queued in
        // this batch window. Steering away from them ensures no two descents
        // ever land on the same unexpanded node, eliminating collisions without
        // any retry overhead.
        double sqrt_parent_n = std::sqrt(static_cast<double>(std::max(1, node->N)));
        double best_score = -1e18;
        Node* best_child = nullptr;
        for (auto& kv : node->children) {
            Node* c = kv.second.get();
            if (c->in_flight) continue;  // already queued this batch, steer away
            double q = (c->N > 0) ? -(c->W / c->N) : 0.0;
            double u = C_PUCT * c->P * sqrt_parent_n / (1.0 + c->N);
            if (q + u > best_score) { best_score = q + u; best_child = c; }
        }
        // All children in_flight (only when branching factor <= BATCH_SIZE and
        // all slots filled): fall back without the skip so the sim can proceed.
        if (!best_child) {
            for (auto& kv : node->children) {
                Node* c = kv.second.get();
                double q = (c->N > 0) ? -(c->W / c->N) : 0.0;
                double u = C_PUCT * c->P * sqrt_parent_n / (1.0 + c->N);
                if (q + u > best_score) { best_score = q + u; best_child = c; }
            }
        }
        best_child->N += VIRTUAL_LOSS;
        best_child->W += VIRTUAL_LOSS;
        node = best_child;
        path.push_back(node);
    }
}

// Back up a completed sim.
//
// BUG FIX vs previous batching attempt: the leaf (last index) DOES receive
// virtual loss (it was selected as a best_child during the final PUCT step),
// so we undo it here -- same as original simulate_one() which undoes VL for
// every node except index 0 (root).
static void backup(const std::vector<Node*>& path, double leaf_value) {
    double v = leaf_value;
    for (int i = (int)path.size() - 1; i >= 0; --i) {
        Node* n = path[i];
        if (i != 0) {
            // Undo the virtual loss applied when this node was selected
            // as a best_child during descent (root was never selected).
            n->N -= VIRTUAL_LOSS;
            n->W -= VIRTUAL_LOSS;
        }
        n->N += 1;
        n->W += v;
        v = -v;
    }
}

// Run `sims` simulations against `root`, batching NN leaf evals
// BATCH_SIZE at a time to amortise pipe round-trip overhead.
//
// Correctness properties (matching original simulate_one behaviour):
//   - VIRTUAL_LOSS=3 is applied and properly undone on every path.
//   - in_flight collisions are retried, never backed up with 0.0.
//   - Terminal and empty-expanded leaves resolve inline (no batch slot).
static void run_sims(Node* root, int sims) {
    int completed = 0;

    while (completed < sims) {
        std::vector<PendingSim> pending;
        int to_collect = std::min(BATCH_SIZE, sims - completed);

        // Collect up to BATCH_SIZE leaf evals.
        // Terminals resolve inline (completed++).
        // Non-terminals are parked in pending for the batch request.
        // Collisions are eliminated by PUCT skipping in_flight children,
        // so descend() always returns either resolved=true or a valid leaf.
        while ((int)pending.size() < to_collect && completed + (int)pending.size() < sims) {
            PendingSim ps;
            bool resolved = descend(root, ps);
            if (resolved) {
                completed++;
            } else {
                pending.push_back(std::move(ps));
            }
        }

        if (pending.empty()) continue;

        // Build and send the batch request.
        std::ostringstream oss;
        oss << "{\"cmd\":\"batch\",\"requests\":[";
        for (size_t i = 0; i < pending.size(); ++i) {
            if (i) oss << ',';
            oss << json_write(pending[i].request);
        }
        oss << "]}";

        std::string resp_line;
        {
            std::lock_guard<std::mutex> io(g_io_mutex);
            std::cout << oss.str() << std::endl;
            if (!std::getline(std::cin, resp_line)) {
                std::cerr << "[mcts_engine] FATAL: stdin closed awaiting batch_result" << std::endl;
                std::exit(1);
            }
        }

        // Parse batch_result.
        JsonObj outer = json_parse(resp_line);
        const std::vector<std::string>& raw_resps =
            outer.count("responses") ? outer["responses"].arr_s
                                     : std::vector<std::string>{};

        // Apply responses and backup.
        for (size_t i = 0; i < pending.size(); ++i) {
            JsonObj resp = (i < raw_resps.size())
                ? json_parse(raw_resps[i])
                : [&]() {
                    // Fallback: treat as draw if response missing.
                    JsonObj safe;
                    safe["terminal"] = JsonVal::Bool(true);
                    safe["result"]   = JsonVal::Num(0.0);
                    safe["value"]    = JsonVal::Num(0.0);
                    safe["moves"]    = JsonVal::ArrStr({});
                    safe["priors"]   = JsonVal::ArrNum({});
                    return safe;
                  }();

            double leaf_val = apply_response(pending[i].leaf, root, resp);
            backup(pending[i].path, leaf_val);
            completed++;
        }
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

            // Evaluate root first with a dedicated single call -- it must
            // complete before any sims start and there is nothing to batch.
            {
                JsonObj root_req;
                root_req["type"] = JsonVal::Str("root");
                root_req["history"] = JsonVal::ArrStr(root->history);
                JsonObj root_resp = send_request(root_req);
                apply_response(root.get(), root.get(), root_resp);
            }

            // Run all sims, batching BATCH_SIZE leaf evals per round-trip.
            // All simulations run on THIS thread (the main thread that owns
            // stdin/stdout). Running them in a thread pool was a deadlock:
            // each sim calls blocking I/O, but main() was also blocked on
            // std::getline. Serial execution here is correct.
            run_sims(root.get(), sims);

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
