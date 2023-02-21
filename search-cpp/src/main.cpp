#include <cmath>
#include <iostream>
#include <ranges>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <unordered_set>
#include <queue>
#include <random>
#include <functional>
#include <algorithm>
#include <fmt/format.h>
#include <fmt/ranges.h>
using namespace std;

static inline constexpr uint64_t lg_down(uint64_t x) {
  return 63U - __builtin_clzl(x);
}

/* base-2 logarithm, rounding up */
static inline constexpr uint64_t lg_up(uint64_t x) {
  return lg_down(x - 1) + 1;
}


template<uint32_t _N>
struct State {
    static_assert(_N <= 4, "uint64_t only support 4 * 4");
enum : uint32_t {
       N = _N,
       BW = lg_up(N * N),
       U = (((1LL) << BW) - 1)
    };
    using CompressT = uint64_t;
    using DecodeT = array<uint8_t, N * N>;
    CompressT st = 0;
    int32_t prev = -1;
    int32_t depth = 0;
    static auto indexOf(CompressT st) {
        for(int i = 0; i < N * N; ++i) {
            if(((st >> (i * BW)) & U) == 0)
                return i;
        }
        return -1;
    }
    static auto decode(CompressT st) -> DecodeT {
        // views::iota(0, N * N) | views::transform([st](int i) { return ((st >> (i * BW)) & U);});
        DecodeT ans;
        for(int i = 0; i < N * N; ++i) {
            ans[i] = (st >> (i * BW)) & U;
        }
        return ans;
    }
    static auto encode(const DecodeT& code) -> CompressT {
        // enumrate(code) | transform([](i, v){ return v << (i * BW); }) | reduce([](l, r){ return l | r; })
        CompressT ans = 0;
        for(int i = 0; i < code.size(); ++i) {
            ans |= ((uint64_t)code[i]) << (i * BW);
        }
        return ans;
    }
};

mt19937 rng{std::random_device{}()};

template<uint32_t _N, bool isRandom, bool isAstar, bool hType>
struct BfsSolver {
    enum { N = _N };
    using i32 = int32_t;
    using ST = State<N>;
    using CompressT = typename ST::CompressT;
    using DecodeT = typename ST::DecodeT;
    unordered_set<CompressT> vis;
    struct QT {
        CompressT st;
        i32 depth;
        i32 h;
        friend bool operator < (const QT&l, const QT& r) {
            return l.h > r.h;
        }
    };
    priority_queue<QT> q;
    CompressT init_state = -1, target_state = -1;
    typename ST::DecodeT tard;
    i32 max_depth = 0;
    auto with_state(CompressT _init_state, CompressT _target_state) -> BfsSolver& {
        vis.clear();
        init_state = _init_state;
        target_state = _target_state;
        max_depth = 0;
        vis.insert(_init_state);
        auto tmp = ST::decode(target_state);
        for(int i = 0; i < N * N; ++i) {
            tard[tmp[i]] = i;
        }
        return *this;
    }
    auto with_h(function<i32(DecodeT, DecodeT)> h) {
        this->h = h;
        return *this;
    }

    auto get_h(CompressT st) -> i32 {
        static typename ST::DecodeT t;
        t = ST::decode(st);
        i32 ans = 0;
        if constexpr (hType == 0) { // h1
            for(int i = 0; i < N * N; ++i) {
                ans += (abs(i / N - tard[t[i]] / N) + abs(i % N - tard[t[i]] % N));
            }
        } else { // h2
            for(int i = 0; i < N * N; ++i) {
                ans += 2 * max(abs(i / N - tard[t[i]] / N), abs(i % N - tard[t[i]] % N));
            }
        }
        if constexpr (isRandom) {
            ans = ans * 10 + rng() % 11;
        }
        return ans;
    }

    auto get_next_st(CompressT st) -> vector<CompressT> {
        static i32 dx[] = {0, -1, 0, 1};
        static i32 dy[] = {-1, 0, 1, 0};
        vector<CompressT> ans;
        i32 idx = ST::indexOf(st);
        typename ST::DecodeT tmp;
        for(int k = 0; k < 4; ++k) {
            i32 tx = (idx / N) + dx[k], ty = (idx % N) + dy[k];
            if (tx >= 0 && tx < N && ty >= 0 && ty < N) {
                tmp = ST::decode(st);
                swap(tmp[idx], tmp[tx * N + ty]);
                CompressT t = ST::encode(tmp);
                if(vis.count(t) == 0) {
                    ans.push_back(t);
                    vis.insert(t);
                }
            }
        }
        return ans;
    }

    auto do_search(CompressT st) -> bool {
        vis.insert(st);
        q.push(QT{st, 0, 0});
        if (st == target_state) return true;
        while(!q.empty()) {
            auto t = q.top();
            q.pop();
            auto next_st = get_next_st(t.st);
            max_depth = max(max_depth, t.depth + 1);
            for(auto n: next_st) {
                if (n == target_state) return true;
                // fmt::print("search:{} -> {}\n", ST::decode(t.st), ST::decode(n));
                QT qt{n, t.depth + 1, 0};
                if constexpr (isAstar) {
                    qt.h = get_h(n);
                } else if constexpr(isRandom){
                    qt.h = rng() % 10000000;
                }
                q.push(qt);
            }
        }
        return false;
    }
    auto search() -> BfsSolver& {
        if(!do_search(init_state)) {
            fmt::print("search faild, check input!!!\n");
        }
        return *this;
    }
    auto summary() -> pair<i32, i32> {
        fmt::print("summary:\nstate count:{}\nmax depth:{}\n", vis.size(), max_depth);
        return make_pair<i32, i32>((i32)vis.size(), (i32)max_depth);
    }
};

template<uint32_t _N, bool isRandom, bool isAstar, bool hType>
struct DfsSolver {
    enum { N = _N };
    using i32 = int32_t;
    using ST = State<N>;
    using CompressT = typename ST::CompressT;
    using DecodeT = typename ST::DecodeT;
    unordered_set<CompressT> vis;
    CompressT init_state = -1, target_state = -1;
    typename ST::DecodeT tard;
    i32 max_depth = 0;
    auto with_state(CompressT _init_state, CompressT _target_state) -> DfsSolver& {
        vis.clear();
        init_state = _init_state;
        target_state = _target_state;
        max_depth = 0;
        vis.insert(_init_state);
        auto tmp = ST::decode(target_state);
        for(int i = 0; i < N * N; ++i) {
            tard[tmp[i]] = i;
        }
        return *this;
    }
    auto with_h(function<i32(DecodeT, DecodeT)> h) {
        this->h = h;
        return *this;
    }

    auto get_h(CompressT st) -> i32 {
        static typename ST::DecodeT t;
        t = ST::decode(st);
        i32 ans = 0;
        if constexpr (hType == 0) { // simple h
            for(int i = 0; i < N * N; ++i) {
                ans += (abs(i / N - tard[t[i]] / N) + abs(i % N - tard[t[i]] % N));
            }
        } else {
            for(int i = 0; i < N * N; ++i) {
                ans += 2 * max(abs(i / N - tard[t[i]] / N), abs(i % N - tard[t[i]] % N));
            }
        }
        return ans;
    }

    auto astar_shuffle(vector<CompressT>&ans) -> void {
        vector<pair<CompressT, i32>> tmp;
        for(auto st: ans) {
            i32 h = get_h(st);
            if constexpr (isRandom) {
                h = h * 10 + abs((int)rng() % 11);
            }
            tmp.push_back(make_pair(st, h));
        }
        sort(tmp.begin(), tmp.end(), [](const pair<CompressT, i32> & l, const pair<CompressT, i32>& r) {return l.second < r.second; });
        for(int i = 0; i < tmp.size(); ++i) {
            ans[i] = tmp[i].first;
        }
    }

    auto get_next_st(CompressT st) -> vector<CompressT> {
        static i32 dx[] = {0, -1, 0, 1};
        static i32 dy[] = {-1, 0, 1, 0};
        vector<CompressT> ans;
        i32 idx = ST::indexOf(st);
        typename ST::DecodeT tmp;
        for(int k = 0; k < 4; ++k) {
            i32 tx = (idx / N) + dx[k], ty = (idx % N) + dy[k];
            if (tx >= 0 && tx < N && ty >= 0 && ty < N) {
                tmp = ST::decode(st);
                swap(tmp[idx], tmp[tx * N + ty]);
                CompressT t = ST::encode(tmp);
                if(vis.count(t) == 0) {
                    ans.push_back(t);
                    vis.insert(t);
                }
            }
        }

        if constexpr (isAstar) {
            astar_shuffle(ans);
        } else if constexpr (isRandom) {
            shuffle(ans.begin(), ans.end(), rng);
        }

        return ans;
    }
    auto do_search(CompressT st, i32 depth) -> bool {
        // fmt::print("searching: {}\n", ST::decode(st));
        max_depth = max(max_depth, depth);
        if(st == target_state) {
            fmt::print("search succeed!\nshowing back trace\n{}:{}\n", depth, ST::decode(st));
            return true;
        }
        auto rs = get_next_st(st);
        for(auto i : rs) {
            if(do_search(i, depth + 1)) {
                // fmt::print("{}:{}\n", depth, st);
                return true;
            }
        }
        return false;
    }
    auto search() -> DfsSolver& {
        if(!do_search(init_state, 0)) {
            fmt::print("search faild, check input!!!\n");
        }
        return *this;
    }
    auto summary() -> pair<i32, i32> {
        fmt::print("summary:\nstate count:{}\nmax depth:{}\n", vis.size(), max_depth);
        return make_pair<i32, i32>((i32)vis.size(), (i32)max_depth);
    }
};


void dfs_test3() {
    State<3>::DecodeT init_state{2, 8, 3, 1, 6, 4, 7, 0, 5}, tar{1, 2, 3, 8, 0, 4, 7, 6, 5};
    for(int i = 0; i < 10; ++i) {
        fmt::print("==================================================\n");
        fmt::print("==================dfs test3 round:{}==============\n", i);
        fmt::print("==================================================\n");
        shuffle(init_state.begin(), init_state.end(), rng);
        shuffle(tar.begin(), tar.end(), rng);
        fmt::print("++++++++++++\ninit:{}\ntarget:{}\n+++++++++++++\n", init_state, tar);
        fmt::print("========== no random =============================\n");
        DfsSolver<3, false, false, false>{}.with_state(State<3>::encode(init_state), State<3>::encode(tar)).search().summary();
        fmt::print("========== random =============================\n");
        DfsSolver<3, true, false, false>{}.with_state(State<3>::encode(init_state), State<3>::encode(tar)).search().summary();
        fmt::print("========== astar + h1 =============================\n");
        DfsSolver<3, false, true, false>{}.with_state(State<3>::encode(init_state), State<3>::encode(tar)).search().summary();
        fmt::print("========== astar + random + h1 =============================\n");
        DfsSolver<3, true, true, false>{}.with_state(State<3>::encode(init_state), State<3>::encode(tar)).search().summary();
        fmt::print("========== astar + h2 =============================\n");
        DfsSolver<3, false, true, true>{}.with_state(State<3>::encode(init_state), State<3>::encode(tar)).search().summary();
        fmt::print("========== astar + random + h2 =============================\n");
        DfsSolver<3, true, true, true>{}.with_state(State<3>::encode(init_state), State<3>::encode(tar)).search().summary();
    }
}

void bfs_test3() {
    State<3>::DecodeT init_state{2, 8, 3, 1, 6, 4, 7, 0, 5}, tar{1, 2, 3, 8, 0, 4, 7, 6, 5};
    for(int i = 0; i < 10; ++i) {
        fmt::print("==================================================\n");
        fmt::print("==================bfs test3 round:{}==============\n", i);
        fmt::print("==================================================\n");
        shuffle(init_state.begin(), init_state.end(), rng);
        shuffle(tar.begin(), tar.end(), rng);
        fmt::print("++++++++++++\ninit:{}\ntarget:{}\n+++++++++++++\n", init_state, tar);
        fmt::print("========== no random =============================\n");
        BfsSolver<3, false, false, false>{}.with_state(State<3>::encode(init_state), State<3>::encode(tar)).search().summary();
        fmt::print("========== random =============================\n");
        BfsSolver<3, true, false, false>{}.with_state(State<3>::encode(init_state), State<3>::encode(tar)).search().summary();
        fmt::print("========== astar + h1 =============================\n");
        BfsSolver<3, false, true, false>{}.with_state(State<3>::encode(init_state), State<3>::encode(tar)).search().summary();
        fmt::print("========== astar + random + h1 =============================\n");
        BfsSolver<3, true, true, false>{}.with_state(State<3>::encode(init_state), State<3>::encode(tar)).search().summary();
        fmt::print("========== astar + h2 =============================\n");
        BfsSolver<3, false, true, true>{}.with_state(State<3>::encode(init_state), State<3>::encode(tar)).search().summary();
        fmt::print("========== astar + random + h2 =============================\n");
        BfsSolver<3, true, true, true>{}.with_state(State<3>::encode(init_state), State<3>::encode(tar)).search().summary();
    }
}

vector<State<4>::DecodeT> tar_arr {
    {9, 3, 12, 13, 6, 7, 14, 0, 1, 5, 8, 2, 11, 15, 4, 10 },
    {12, 3, 14, 9, 6, 0, 1, 7, 4, 13, 8, 2, 15, 10, 11, 5},
    {15, 1, 11, 8, 13, 6, 12, 7, 3, 2, 14, 0, 9, 10, 4, 5},
    {3, 14, 5, 2, 8, 4, 12, 15, 11, 9, 6, 10, 1, 0, 7, 13},
    {7, 3, 11, 4, 9, 6, 8, 14, 1, 2, 5, 13, 0, 12, 15, 10},

    {13, 7, 12, 6, 2, 1, 9, 8, 4, 10, 5, 14, 11, 15, 0, 3},
    {15, 8, 2, 1, 13, 9, 7, 3, 11, 14, 12, 4, 6, 0, 10, 5},
    {8, 2, 7, 1, 6, 14, 11, 4, 5, 10, 12, 13, 0, 3, 9, 15},
    {9, 1, 3, 11, 10, 13, 12, 7, 15, 4, 14, 5, 8, 2, 0, 6},
    {10, 11, 7, 3, 6, 2, 9, 14, 15, 12, 5, 8, 4, 0, 13, 1},

    {6, 14, 2, 9, 13, 12, 1, 11, 4, 3, 15, 5, 10, 8, 7, 0},
    {2, 8, 0, 6, 12, 11, 4, 15, 7, 13, 14, 3, 5, 9, 1, 10},
    {12, 11, 6, 3, 2, 14, 9, 8, 10, 0, 13, 7, 4, 5, 15, 1},
    {2, 7, 3, 13, 1, 5, 9, 10, 6, 12, 15, 8, 4, 14, 11, 0},
    {6, 9, 2, 1, 10, 8, 4, 13, 14, 15, 11, 7, 5, 0, 12, 3},

    {14, 1, 5, 11, 8, 13, 2, 10, 4, 12, 0, 15, 3, 6, 9, 7},
    {7, 2, 10, 9, 12, 15, 11, 1, 13, 6, 5, 8, 4, 0, 3, 14},
    {9, 6, 12, 5, 8, 11, 15, 10, 0, 1, 7, 2, 3, 4, 14, 13},
    {5, 13, 2, 9, 6, 15, 10, 8, 7, 3, 0, 4, 11, 14, 12, 1},
    {6, 9, 7, 11, 5, 8, 4, 12, 13, 15, 14, 3, 10, 2, 1, 0},
};
void dfs_test4() {
    State<4>::DecodeT init_state{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    tar{init_state};
    double avg_st[4] = {0, 0, 0, 0};
    double avg_dep[4] = {0, 0, 0, 0};
    for(int i = 0; i < 20; ++i) {
        tar = tar_arr[i];
        fmt::print("==================================================\n");
        fmt::print("==================dfs test4 round:{}==============\n", i);
        fmt::print("==================================================\n");
        fmt::print("++++++++++++\ninit:{}\ntarget:{}\n+++++++++++++\n", init_state, tar);
        fmt::print("========== astar h1 =============================\n");
        auto t = DfsSolver<4, false, true, false>{}.with_state(State<4>::encode(init_state), State<4>::encode(tar)).search().summary();
        avg_st[0] += t.first;
        avg_dep[0] += t.second;
        fmt::print("========== astar + random h1 =============================\n");
        t = DfsSolver<4, true, true, false>{}.with_state(State<4>::encode(init_state), State<4>::encode(tar)).search().summary();
        avg_st[1] += t.first;
        avg_dep[1] += t.second;
        fmt::print("========== astar h2 =============================\n");
        t = DfsSolver<4, false, true, true>{}.with_state(State<4>::encode(init_state), State<4>::encode(tar)).search().summary();
        avg_st[2] += t.first;
        avg_dep[2] += t.second;
        fmt::print("========== astar + random h2 =============================\n");
        t = DfsSolver<4, true, true, true>{}.with_state(State<4>::encode(init_state), State<4>::encode(tar)).search().summary();
        avg_st[3] += t.first;
        avg_dep[3] += t.second;
    }
    for(int i = 0; i < 4; ++i) {
        avg_st[i] /= 20;
        avg_dep[i] /= 20;
    }
    fmt::print("dfs test4: {}, {}\n", avg_st, avg_dep);
}

void bfs_test4() {
    State<4>::DecodeT init_state{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    tar{init_state};
    double avg_st[4] = {0, 0, 0, 0};
    double avg_dep[4] = {0, 0, 0, 0};
    for(int i = 0; i < 20; ++i) {
        tar = tar_arr[i];
        fmt::print("==================================================\n");
        fmt::print("==================bfs test4 round:{}==============\n", i);
        fmt::print("==================================================\n");
        fmt::print("++++++++++++\ninit:{}\ntarget:{}\n+++++++++++++\n", init_state, tar);
        fmt::print("========== astar h1 =============================\n");
        auto t = BfsSolver<4, false, true, false>{}.with_state(State<4>::encode(init_state), State<4>::encode(tar)).search().summary();
        avg_st[0] += t.first;
        avg_dep[0] += t.second;
        fmt::print("========== astar + random h1 =============================\n");
        t = BfsSolver<4, true, true, false>{}.with_state(State<4>::encode(init_state), State<4>::encode(tar)).search().summary();
        avg_st[1] += t.first;
        avg_dep[1] += t.second;
        fmt::print("========== astar h2 =============================\n");
        t = BfsSolver<4, false, true, true>{}.with_state(State<4>::encode(init_state), State<4>::encode(tar)).search().summary();
        avg_st[2] += t.first;
        avg_dep[2] += t.second;
        fmt::print("========== astar + random h2 =============================\n");
        t = BfsSolver<4, true, true, true>{}.with_state(State<4>::encode(init_state), State<4>::encode(tar)).search().summary();
        avg_st[3] += t.first;
        avg_dep[3] += t.second;
    }
    for(int i = 0; i < 4; ++i) {
        avg_st[i] /= 20;
        avg_dep[i] /= 20;
    }
    fmt::print("bfs test4: {}, {}\n", avg_st, avg_dep);
}
int main(int argc, char** argv) {
    bfs_test3();
    dfs_test3();
    bfs_test4();
    dfs_test4();
}
