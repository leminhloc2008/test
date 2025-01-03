#include <bits/stdc++.h>
using namespace std;

/* ========== COPY TOÀN BỘ TEMPLATE Ở TRÊN VÀO ĐÂY ========== */
/*
   (Vì phần quá dài, ở đây giả sử bạn đã copy tất cả struct/hàm 
   BFS, DFS, topoSort, TarjanSCC, Articulation, Floyd, Dijkstra,
   DSU, Kruskal, Dinic, Hopkroft-Karp, Fenwick, Segment Tree, 
   Sparse Table, modExp, matrix exp, Tổ hợp, LCA, Euler Tour, 
   Hash/Z/KMP/Manacher/Trie, LIS/LCS/DP bitmask/Knapsack, 
   Digit DP, DP DAG..., Convex hull, CCW, segment intersect,
   polygon area, point in polygon, backtrack, compress, Mo's, ... )
*/

/* ======== DEMO USAGE CODE (chỉ gọi hàm) ======== */

// 1) Demo BFS/DFS
void demoBFS_DFS() {
    cout << "\n--- Demo BFS/DFS ---\n";
    vector<vector<int>> adj = {
        {1,2},   // 0
        {2,3},   // 1
        {3},     // 2
        {},      // 3
    };
    // BFS from node 0
    auto bfsOrder = BFS(0, adj);
    cout << "BFS order from node 0: ";
    for (auto &x : bfsOrder) cout << x << " ";
    cout << "\n";
    // DFS from node 0
    auto dfsOrder = DFS(0, adj);
    cout << "DFS order from node 0: ";
    for (auto &x : dfsOrder) cout << x << " ";
    cout << "\n";
}

// 2) Demo Topo Sort (Kahn)
void demoTopo() {
    cout << "\n--- Demo Topological Sort ---\n";
    // 4 nodes (0 -> 1 -> 2, 0 -> 2, 2 -> 3)
    vector<vector<int>> adj = {
        {1,2}, // 0
        {2},   // 1
        {3},   // 2
        {}     // 3
    };
    auto order = topoSort(adj);
    cout << "Topo sort: ";
    for (auto &x : order) cout << x << " ";
    cout << "\n";
}

// 3) Demo SCC (Tarjan)
void demoSCC() {
    cout << "\n--- Demo Tarjan SCC ---\n";
    /*
       Giả sử đồ thị có 5 đỉnh: 0->1, 1->2, 2->0, 1->3, 3->4
       => SCC {0,1,2}, {3}, {4}
    */
    TarjanSCC scc(5);
    scc.addEdge(0,1);
    scc.addEdge(1,2);
    scc.addEdge(2,0);
    scc.addEdge(1,3);
    scc.addEdge(3,4);
    scc.buildSCC();
    cout << "SCC comp array:\n";
    for(int i=0;i<5;i++){
        cout << "Node " << i << " in SCC " << scc.comp[i] << "\n";
    }
}

// 4) Demo Articulation points & Bridges
void demoArticulationAndBridges() {
    cout << "\n--- Demo Articulation & Bridges ---\n";
    Articulation art(5);
    /*
       0--1--2
       |  |
       3--4
       => 1 là khớp? Tùy cấu trúc. Ta chỉ minh họa.
    */
    art.addEdge(0,1);
    art.addEdge(1,2);
    art.addEdge(1,3);
    art.addEdge(3,4);
    art.addEdge(4,1);
    art.build();
    cout << "Articulation points: ";
    for (auto &x : art.articulationPoints) {
        cout << x << " ";
    }
    cout << "\nBridges:\n";
    for (auto &b : art.bridges) {
        cout << b.first << "-" << b.second << "\n";
    }
}

// 5) Demo Floyd & Dijkstra
void demoFloyd_Dijkstra() {
    cout << "\n--- Demo Floyd Warshall & Dijkstra ---\n";
    // Giả sử graph 4 nodes, adjacency matrix cho Floyd
    const long long INF = LLONG_MAX;
    vector<vector<long long>> dist = {
        {0,   3, INF, 5  },
        {2,   0,   8, INF},
        {INF, INF, 0,   1},
        {INF, INF, INF, 0},
    };
    floydWarshall(dist);
    cout << "Floyd distance matrix:\n";
    for(int i=0;i<(int)dist.size();i++){
        for(int j=0;j<(int)dist.size();j++){
            if(dist[i][j]==INF) cout << "INF ";
            else cout << dist[i][j] << " ";
        }
        cout << "\n";
    }

    // Demo Dijkstra (list)
    vector<vector<pair<int,int>>> adj = {
       {{1,3},{3,5}},    // node 0 -> (1,3),(3,5)
       {{0,2},{2,8}},    // node 1 -> (0,2),(2,8)
       {{3,1}},          // node 2 -> (3,1)
       {}                // node 3
    };
    auto distDij = dijkstra(0, adj);
    cout << "Dijkstra dist from 0: ";
    for (auto &x : distDij) {
        if(x==LLONG_MAX) cout << "INF ";
        else cout << x << " ";
    }
    cout << "\n";
}

// 6) Demo DSU & MST (Kruskal)
void demoDSU_MST() {
    cout << "\n--- Demo DSU & Kruskal MST ---\n";
    // 5 nodes, edges
    vector<tuple<long long,int,int>> edges = {
        {1, 0,1},
        {4, 1,2},
        {3, 0,2},
        {2, 2,3},
        {10,1,3},
        {5, 3,4}
    };
    long long costMST = kruskal(edges, 5);
    cout << "MST cost = " << costMST << "\n";
}

// 7) Demo Max Flow (Dinic) & Bipartite Matching (Hopkroft-Karp)
void demoFlows() {
    cout << "\n--- Demo Dinic MaxFlow ---\n";
    // Graph (4 nodes: s=0, t=3)
    Dinic dinic(4, 0, 3);
    dinic.addEdge(0,1,10);
    dinic.addEdge(0,2,5);
    dinic.addEdge(1,2,15);
    dinic.addEdge(1,3,10);
    dinic.addEdge(2,3,10);
    cout << "Max flow = " << dinic.maxFlow() << "\n";

    cout << "\n--- Demo Hopkroft-Karp ---\n";
    // Bipartite: U={1..3}, V={1..4}
    HopkroftKarp hk(3, 4);
    /*
       U1 -> V2, V3
       U2 -> V1, V2
       U3 -> V4
    */
    hk.addEdge(1,2);
    hk.addEdge(1,3);
    hk.addEdge(2,1);
    hk.addEdge(2,2);
    hk.addEdge(3,4);
    cout << "Maximum bipartite matching = " << hk.maxMatching() << "\n";
}

// 8) Demo Fenwick & SegmentTree & SparseTable
void demoFenw_Seg_Sparse() {
    cout << "\n--- Demo Fenwick (BIT) ---\n";
    Fenwick fenw(5);
    // update some values
    fenw.update(1, 5); // arr[1] += 5
    fenw.update(3, 2); // arr[3] += 2
    fenw.update(5,10); // arr[5] += 10
    cout << "Fenw query(1..5) = " << fenw.rangeQuery(1,5) << "\n";

    cout << "\n--- Demo Segment Tree ---\n";
    vector<long long> arr = {2,1,3,4,5};
    SegmentTree seg((int)arr.size());
    seg.build(arr);
    cout << "Sum query(0..4) = " << seg.query(0,4) << "\n";
    seg.update(2,10);  // arr[2] = 10
    cout << "Sum query(0..4) after update = " << seg.query(0,4) << "\n";

    cout << "\n--- Demo Sparse Table (RMQ) ---\n";
    vector<long long> arr2 = {2, 1, 3, 4, 5};
    SparseTable st(arr2);
    // query min in [1..3]
    cout << "Min [1..3] = " << st.query(1,3) << "\n";
}

// 9) Demo toán: modExp, matrixExp, Tổ hợp, etc.
void demoMathStuff() {
    cout << "\n--- Demo Math Stuff ---\n";
    // 1) modExp
    cout << "2^10 mod 1000 = " << modExp(2,10,1000) << "\n";
    // 2) Matrix Exp
    matrix base = {
        {1,1},
        {1,0}
    };
    matrix res = matPow(base,5, (long long)1e9+7); 
    cout << "Fibonacci(5) in matrix form top-left = " << res[0][0] << "\n";
    // 3) Tổ hợp
    precomputeFactorial(); // cần gọi 1 lần 
    cout << "C(5,2) mod 1e9+7 = " << C(5,2) << "\n";
    // 4) Bao hàm loại trừ (chỉ demo 1 example)
    cout << "Inclusion-Exclusion example: " << inclusionExclusionExample(30) << "\n";
}

// 10) Demo LCA & Euler Tour
void demoLCA_Euler() {
    cout << "\n--- Demo LCA & Euler Tour ---\n";
    // Tree: 0-1, 0-2, 2-3
    LCA lcaTool(4);
    lcaTool.addEdge(0,1);
    lcaTool.addEdge(0,2);
    lcaTool.addEdge(2,3);
    // Dfs(0,-1)
    lcaTool.dfs(0,-1);
    cout << "LCA(1,3) = " << lcaTool.lca(1,3) << "\n";

    // Euler Tour
    vector<vector<int>> adj = {
        {1,2}, //0
        {},    //1
        {3},   //2
        {}     //3
    };
    eulerIn.resize(4); eulerOut.resize(4);
    eulerTourArr.clear(); 
    timerEuler = 0;
    eulerTourDFS(0,-1, adj);
    cout << "Euler Tour order: ";
    for (auto &x : eulerTourArr) cout << x << " ";
    cout << "\n";
}

// 11) Demo String stuff: Hash, Z, KMP, Manacher, Trie
void demoString() {
    cout << "\n--- Demo String Stuff ---\n";
    string s="ababa";
    // 1) Hash
    StringHash sh(s, 137, 1000000007);
    cout << "Hash(0..3) => [s.substr(0,3)='aba'] = " << sh.getHash(0,3) << "\n";
    // 2) Z-function
    auto z=Z_function(s);
    cout << "Z array of 'ababa': ";
    for (auto &x: z) cout << x << " ";
    cout << "\n";
    // 3) prefix function (KMP)
    auto pi=prefix_function(s);
    cout << "pi array of 'ababa': ";
    for (auto &x: pi) cout << x << " ";
    cout << "\n";
    // 4) Manacher
    auto d=manacherOdd(s);
    cout << "Manacher odd radius: ";
    for (auto &x: d) cout << x << " ";
    cout << "\n";
    // 5) Trie
    Trie tr;
    tr.insert("apple");
    tr.insert("app");
    cout << "Trie search 'app'? " << tr.search("app") << "\n"; 
    cout << "Trie search 'banana'? " << tr.search("banana") << "\n";
}

// 12) Demo DP: LIS, LCS, DP bitmask, Knapsack, Digit DP, DP on DAG...
void demoDP() {
    cout << "\n--- Demo DP ---\n";
    // 1) LIS
    vector<int> arr = {3,4,-1,5,8,2,3,12};
    cout << "LIS length = " << LIS(arr) << "\n";
    // 2) LCS
    cout << "LCS('abcde','ace') = " << LCS("abcde","ace") << "\n";
    // 3) DP bitmask
    // (chỉ minh hoạ 1 table w, cost= i*j ?)
    vector<vector<int>> w(3, vector<int>(3,0));
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            w[i][j] = i*j; 
        }
    }
    cout << "DP bitmask matching cost = " << dpBitmaskMatching(w) << "\n";
    // 4) Knapsack
    vector<int> wt={1,3,4}, val={15,20,30};
    cout << "Knapsack best = " << knapsack(wt,val,4) << "\n";
    // 5) Digit DP
    cout << "Digit DP example, count up to 13? => " << digitDP(13) << "\n";
    // 6) DP DAG (longest path)
    vector<vector<int>> graph = {
        {1,2}, //0
        {3},   //1
        {3},   //2
        {}     //3
    };
    auto topo = topoSort(graph);
    auto distDAG = DAGLongestPath(graph, topo);
    cout << "Longest path distance from node 0 to 3 = " << distDAG[3] << "\n";
}

// 13) Demo hình học: convex hull, ccw, intersect, polygon area, point in polygon
void demoGeometry() {
    cout << "\n--- Demo Geometry ---\n";
    vector<pt> pts = {{0,0},{1,0},{2,2},{1,1},{0,2}};
    auto hull = convexHull(pts);
    cout << "Convex hull points: ";
    for (auto &p : hull) {
        cout << "(" << p.x << "," << p.y << ") ";
    }
    cout << "\nPolygon area of hull = " << polygonArea(hull) << "\n";

    // ccw
    pt A{0,0}, B{1,1}, C{2,2};
    cout << "CCW( (0,0)->(1,1)->(2,2) ) = " << ccw(A,B,C) << "\n";

    // segment intersect
    pt D{1,1}, E{1,2};
    cout << "SegmentIntersect(AB,DE)? " << segmentIntersect(A,B,D,E) << "\n";

    // point in polygon
    pt P{1,1};
    cout << "Point(1,1) in polygon? " << pointInPolygon(P, hull) << "\n";
}

// 14) Demo Backtracking, compress, Mo's 
void demoOthers() {
    cout << "\n--- Demo Backtracking ---\n";
    // Generate all binary strings of length 3
    int n=3;
    string temp(n,'0');
    cout << "All binary strings of length 3:\n";
    function<void(int)> backtrack = [&](int idx){
        if(idx==n){
            cout << temp << "\n";
            return;
        }
        for(char c='0'; c<='1'; c++){
            temp[idx] = c;
            backtrack(idx+1);
        }
    };
    backtrack(0);

    cout << "\n--- Demo Coordinate Compression ---\n";
    vector<int> arr = {10,100,10,50,20};
    auto comp = compress(arr);
    cout << "After compress: ";
    for(auto &x: comp) cout << x << " ";
    cout << "\n";

    cout << "\n--- Demo Mo's algorithm (sqrt decomposition) ---\n";
    vector<int> data = {1, 2, 3, 4, 5};
    vector<Query> queries = {
        {0,2,0}, // sum [0..2]
        {1,3,1}, // sum [1..3]
        {2,4,2}  // sum [2..4]
    };
    auto ans = moAlgorithm(data, queries);
    for(int i=0;i<(int)ans.size();i++){
        cout << "Query " << i << " => sum = " << ans[i] << "\n";
    }
}

// =========== MAIN ===========

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    // Gọi tuần tự các demo
    demoBFS_DFS();
    demoTopo();
    demoSCC();
    demoArticulationAndBridges();
    demoFloyd_Dijkstra();
    demoDSU_MST();
    demoFlows();
    demoFenw_Seg_Sparse();
    demoMathStuff();
    demoLCA_Euler();
    demoString();
    demoDP();
    demoGeometry();
    demoOthers();

    return 0;
}
