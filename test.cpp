#include <bits/stdc++.h>
using namespace std;

/* ===============  ĐỒ THỊ  =============== */

/* 1) BFS / DFS cơ bản */
vector<int> BFS(int start, const vector<vector<int>>& adj) {
    // Trả về thứ tự duyệt BFS từ đỉnh 'start'
    int n = (int)adj.size();
    vector<bool> visited(n, false);
    queue<int>q;
    vector<int>order;
    
    visited[start] = true;
    q.push(start);

    while(!q.empty()) {
        int u = q.front(); 
        q.pop();
        order.push_back(u);
        for (auto &v : adj[u]) {
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);
            }
        }
    }
    return order;
}

void DFS_util(int u, const vector<vector<int>>& adj, vector<bool>& visited, vector<int>& order) {
    visited[u] = true;
    order.push_back(u);
    for (auto &v : adj[u]) {
        if (!visited[v]) {
            DFS_util(v, adj, visited, order);
        }
    }
}

vector<int> DFS(int start, const vector<vector<int>>& adj) {
    // Trả về thứ tự duyệt DFS từ đỉnh 'start'
    int n = (int)adj.size();
    vector<bool> visited(n, false);
    vector<int>order;
    DFS_util(start, adj, visited, order);
    return order;
}

/* 2) Topological Sort (Kahn's Algorithm) */
vector<int> topoSort(const vector<vector<int>>& adj) {
    int n = (int)adj.size();
    vector<int> inDeg(n, 0), order;
    for(int u = 0; u < n; u++){
        for(auto &v : adj[u]){
            inDeg[v]++;
        }
    }
    queue<int>q;
    for(int i = 0; i < n; i++){
        if(inDeg[i] == 0) q.push(i);
    }
    while(!q.empty()){
        int u = q.front(); q.pop();
        order.push_back(u);
        for(auto &v : adj[u]){
            inDeg[v]--;
            if(inDeg[v] == 0) q.push(v);
        }
    }
    // Nếu muốn kiểm tra có phải DAG không, kiểm tra size(order) == n
    return order;
}

/* 3) Tìm thành phần liên thông mạnh (SCC) - Tarjan */
struct TarjanSCC {
    int n, timeDFS;
    vector<int> low, num, comp; 
    // comp[u] sẽ cho biết SCC-index của đỉnh u
    vector<bool> inStack;
    stack<int> st;
    vector<vector<int>> adj;

    TarjanSCC(int n): n(n), adj(n) {
        low.assign(n, 0); 
        num.assign(n, -1); 
        comp.assign(n, -1);
        inStack.assign(n, false);
        timeDFS = 0;
    }
    void addEdge(int u, int v){
        adj[u].push_back(v);
    }
    void dfs(int u){
        low[u] = num[u] = timeDFS++;
        st.push(u); 
        inStack[u] = true;
        for(int v : adj[u]){
            if(num[v] == -1){
                dfs(v);
                low[u] = min(low[u], low[v]);
            } else if(inStack[v]) {
                low[u] = min(low[u], num[v]);
            }
        }
        // root SCC
        if(low[u] == num[u]){
            // lấy ra các đỉnh thuộc SCC này
            while(true){
                int v = st.top(); 
                st.pop();
                inStack[v] = false;
                comp[v] = u; // gán cha đại diện
                if(v == u) break;
            }
        }
    }
    void buildSCC(){
        for(int i = 0; i < n; i++){
            if(num[i] == -1) dfs(i);
        }
    }
};

/* 4) Khớp và Cầu (Articulation Points & Bridges) */
struct Articulation {
    int n, t;
    vector<vector<int>> adj;
    vector<int> tin, low;
    vector<bool> visited;
    set<int> articulationPoints;
    vector<pair<int,int>> bridges;

    Articulation(int n):n(n),adj(n), tin(n), low(n), visited(n,false){
        t = 0;
    }
    void addEdge(int u,int v){
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    void dfs(int u,int p = -1){
        visited[u] = true;
        tin[u] = low[u] = t++;
        int children = 0;
        for(auto &v: adj[u]){
            if(v == p) continue;
            if(!visited[v]){
                dfs(v,u);
                low[u] = min(low[u], low[v]);
                if(low[v] >= tin[u] && p != -1){
                    articulationPoints.insert(u);
                }
                if(low[v] > tin[u]){
                    bridges.push_back({u,v});
                }
                children++;
            } else {
                low[u] = min(low[u], tin[v]);
            }
        }
        if(p == -1 && children > 1){
            articulationPoints.insert(u);
        }
    }
    void build(){
        for(int i=0; i<n; i++){
            if(!visited[i]){
                dfs(i);
            }
        }
    }
};

/* 5) Đường đi ngắn nhất (Floyd-Warshall & Dijkstra) */

// Floyd-Warshall O(n^3)
void floydWarshall(vector<vector<long long>>& dist) {
    // dist[i][j] ban đầu là trọng số từ i->j, hoặc 1e15 nếu không có cạnh
    int n = (int)dist.size();
    for(int k=0;k<n;k++){
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                if(dist[i][k] < LLONG_MAX && dist[k][j] < LLONG_MAX)
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
            }
        }
    }
}

// Dijkstra O(m log n)
vector<long long> dijkstra(int start, const vector<vector<pair<int,int>>>& adj){
    // adj[u] = vector of {v, w} (cạnh u->v, trọng số w)
    int n = (int)adj.size();
    vector<long long> dist(n, LLONG_MAX);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<pair<long long,int>>> pq;
    dist[start] = 0;
    pq.push({0,start});
    while(!pq.empty()){
        auto [cd, u] = pq.top(); pq.pop();
        if(cd > dist[u]) continue;
        for(auto &[v, w] : adj[u]){
            long long nd = cd + w;
            if(nd < dist[v]){
                dist[v] = nd;
                pq.push({nd,v});
            }
        }
    }
    return dist;
}

/* 6) DSU + Cây khung nhỏ nhất (Kruskal) */
struct DSU {
    vector<int> par, sz;
    DSU(int n){
        par.resize(n);
        sz.resize(n,1);
        for(int i=0;i<n;i++) par[i] = i;
    }
    int findp(int v){
        if(par[v] == v) return v;
        return par[v] = findp(par[v]);
    }
    bool unite(int a,int b){
        a = findp(a); b = findp(b);
        if(a == b) return false;
        if(sz[a] < sz[b]) swap(a,b);
        par[b] = a;
        sz[a]+=sz[b];
        return true;
    }
};

// Kruskal
long long kruskal(vector<tuple<long long,int,int>>& edges, int n){
    // edges: vector {weight, u, v}
    // n: số đỉnh
    // return tổng trọng lượng MST, nếu đồ thị không liên thông có thể xử lý tùy
    sort(edges.begin(), edges.end());
    DSU dsu(n);
    long long mstCost = 0;
    for(auto &[w, u, v] : edges){
        if(dsu.unite(u,v)){
            mstCost += w;
        }
    }
    return mstCost;
}

/* 7) Luồng cực đại (Dinic) */
struct Dinic {
    struct Edge {
        int to, rev;
        long long cap;
    };
    vector<vector<Edge>> adj;
    vector<int> level, it;
    int n, s, t;

    Dinic(int n, int s, int t) : n(n), s(s), t(t) {
        adj.assign(n, {});
        level.assign(n, -1);
        it.assign(n, 0);
    }
    void addEdge(int u,int v,long long c){
        adj[u].push_back({v,(int)adj[v].size(), c});
        adj[v].push_back({u,(int)adj[u].size()-1, 0});
    }
    bool bfs(){
        fill(level.begin(), level.end(), -1);
        level[s] = 0;
        queue<int>q; q.push(s);
        while(!q.empty()){
            int u = q.front();q.pop();
            for(auto &e : adj[u]){
                if(level[e.to] == -1 && e.cap > 0){
                    level[e.to] = level[u]+1;
                    q.push(e.to);
                }
            }
        }
        return level[t] != -1;
    }
    long long sendFlow(int u, long long f){
        if(!f) return 0;
        if(u == t) return f;
        for(int &cid=it[u]; cid<(int)adj[u].size(); cid++){
            auto &e = adj[u][cid];
            if(e.cap > 0 && level[e.to] == level[u]+1){
                long long cur = sendFlow(e.to, min(f, e.cap));
                if(cur){
                    e.cap -= cur;
                    adj[e.to][e.rev].cap += cur;
                    return cur;
                }
            }
        }
        return 0;
    }
    long long maxFlow(){
        long long total = 0;
        while(bfs()){
            fill(it.begin(), it.end(), 0);
            while(long long f = sendFlow(s, LLONG_MAX)) {
                total += f;
            }
        }
        return total;
    }
};

/* 8) Cặp ghép cực đại trên đồ thị hai phía (Hopkroft-Karp) */
struct HopkroftKarp {
    int n, m; // n đỉnh tập U, m đỉnh tập V
    vector<vector<int>> adj; // adj[u] = list các v kết nối
    vector<int> dist, matchU, matchV;

    HopkroftKarp(int n, int m): n(n), m(m) {
        adj.resize(n+1);
        dist.resize(n+1);
        matchU.resize(n+1, 0);
        matchV.resize(m+1, 0);
    }
    void addEdge(int u, int v){
        // u thuộc [1..n], v thuộc [1..m]
        adj[u].push_back(v);
    }
    bool bfs(){
        queue<int>q;
        for(int i=1;i<=n;i++){
            if(!matchU[i]){
                dist[i] = 0;
                q.push(i);
            } else dist[i] = INT_MAX;
        }
        dist[0] = INT_MAX;
        while(!q.empty()){
            int u = q.front(); q.pop();
            if(dist[u] < dist[0]){
                for(auto &v: adj[u]){
                    if(dist[matchV[v]] == INT_MAX){
                        dist[matchV[v]] = dist[u] + 1;
                        q.push(matchV[v]);
                    }
                }
            }
        }
        return dist[0] != INT_MAX;
    }
    bool dfs(int u){
        if(u){
            for(auto &v: adj[u]){
                if(dist[matchV[v]] == dist[u] + 1 && dfs(matchV[v])){
                    matchV[v] = u;
                    matchU[u] = v;
                    return true;
                }
            }
            dist[u] = INT_MAX;
            return false;
        }
        return true;
    }
    int maxMatching(){
        int matching = 0;
        while(bfs()){
            for(int i=1;i<=n;i++){
                if(!matchU[i] && dfs(i)) matching++;
            }
        }
        return matching;
    }
};

/* ===============  CẤU TRÚC DỮ LIỆU  =============== */

/* Fenwick Tree (BIT) */
struct Fenwick {
    int n;
    vector<long long> fenw;
    Fenwick(int n):n(n),fenw(n+1,0){}
    void update(int i,long long v){
        for(; i<=n; i+=(i & -i)){
            fenw[i]+=v;
        }
    }
    long long query(int i){
        long long s=0;
        for(;i>0;i-=(i&-i)){
            s+=fenw[i];
        }
        return s;
    }
    long long rangeQuery(int l,int r){
        return query(r) - query(l-1);
    }
};

/* Segment Tree (min/max/sum) - ví dụ sum */
struct SegmentTree {
    int n;
    vector<long long> seg;
    SegmentTree(int n){
        this->n=n;
        seg.assign(4*n,0LL);
    }
    void build(vector<long long> &a, int idx, int start, int end){
        if(start==end){
            seg[idx]=a[start];
            return;
        }
        int mid=(start+end)/2;
        build(a, idx*2, start, mid);
        build(a, idx*2+1, mid+1, end);
        seg[idx]=seg[idx*2]+seg[idx*2+1];
    }
    void build(vector<long long> &a){
        build(a,1,0,n-1);
    }
    void update(int idx, int start, int end, int pos, long long val){
        if(start==end){
            seg[idx]=val;
            return;
        }
        int mid=(start+end)/2;
        if(pos<=mid) update(idx*2, start, mid, pos, val);
        else update(idx*2+1, mid+1, end, pos, val);
        seg[idx]=seg[idx*2]+seg[idx*2+1];
    }
    long long query(int idx, int start, int end, int l, int r){
        if(r<start || l> end) return 0;
        if(l<=start && end<=r) return seg[idx];
        int mid=(start+end)/2;
        return query(idx*2, start, mid, l, r) + query(idx*2+1, mid+1, end, l, r);
    }
    // API
    void update(int pos, long long val){
        update(1,0,n-1, pos, val);
    }
    long long query(int l,int r){
        return query(1,0,n-1,l,r);
    }
};

/* Sparse Table (RMQ) */
struct SparseTable {
    int n, LOG;
    vector<vector<long long>> st;
    vector<int> lg;
    SparseTable(vector<long long> &arr){
        n = (int)arr.size();
        LOG = floor(log2(n)) + 1;
        st.assign(n, vector<long long>(LOG, 0));
        lg.assign(n+1,0);
        for(int i=2;i<=n;i++){
            lg[i] = lg[i/2]+1;
        }
        for(int i=0;i<n;i++){
            st[i][0] = arr[i];
        }
        for(int j=1; j<LOG; j++){
            for(int i=0; i+(1<<j)-1<n; i++){
                st[i][j] = min(st[i][j-1], st[i+(1<<(j-1))][j-1]);
            }
        }
    }
    long long query(int L,int R){
        int j = lg[R-L+1];
        return min(st[L][j], st[R-(1<<j)+1][j]);
    }
};

/* ===============  TOÁN HỌC  =============== */

/* 1) Tính lũy thừa modulo */
long long modExp(long long base, long long exp, long long mod){
    long long res = 1LL;
    base%=mod;
    while(exp>0){
        if(exp & 1) res=(res*base)%mod;
        base=(base*base)%mod;
        exp>>=1;
    }
    return res;
}

/* 2) Nhân ma trận (matrix exponentiation) */
typedef vector<vector<long long>> matrix;
matrix matMul(const matrix &A, const matrix &B, long long mod){
    int n=(int)A.size(), m=(int)B.size(), p=(int)B[0].size();
    // A: n x m, B: m x p
    matrix C(n, vector<long long>(p,0LL));
    for(int i=0;i<n;i++){
        for(int j=0;j<p;j++){
            __int128 sum=0;
            for(int k=0;k<m;k++){
                sum+=(__int128)A[i][k]*B[k][j];
            }
            C[i][j]=(long long)(sum%mod);
        }
    }
    return C;
}
matrix matPow(matrix base, long long exp, long long mod){
    int n=(int)base.size();
    matrix res(n, vector<long long>(n,0LL));
    for(int i=0;i<n;i++){
        res[i][i]=1LL;
    }
    while(exp>0){
        if(exp & 1) res=matMul(res,base,mod);
        base=matMul(base,base,mod);
        exp>>=1;
    }
    return res;
}

/* 3) Tổ hợp (chẳng hạn tiền xử lý factorial, inverse factorial) */
const int MAXN = 200000;
long long fac[MAXN+1], invFac[MAXN+1], MOD=1000000007;
long long inv(long long x){
    // Fermat's little theorem => x^(MOD-2) mod MOD
    return modExp(x,MOD-2,MOD);
}
void precomputeFactorial(){
    fac[0]=1;
    for(int i=1;i<=MAXN;i++){
        fac[i]=(fac[i-1]*1LL*i)%MOD;
    }
    invFac[MAXN] = inv(fac[MAXN]);
    for(int i=MAXN-1;i>=0;i--){
        invFac[i] = (invFac[i+1]*(i+1))%MOD;
    }
}
long long C(int n,int r){
    if(r<0 || r>n) return 0;
    return ((fac[n]*invFac[r])%MOD*invFac[n-r])%MOD;
}

/* 4) Bao hàm loại trừ (Inclusion-Exclusion) */
/*
  Ý tưởng chung:
  S = tổng (f(A)) - tổng (f(giao 2 A)) + tổng(f(giao 3 A)) - ...
  Tùy từng bài cụ thể, triển khai chi tiết.
*/
long long inclusionExclusionExample(int n){
    // Ví dụ đếm số nguyên <= n chia hết bởi ít nhất 1 prime trong {2,3}
    // Count = countDiv(2) + countDiv(3) - countDiv(2*3)
    long long c2 = n/2, c3=n/3, c6=n/6;
    return c2+c3-c6;
}

/* ===============  CÂY  =============== */

/* 1) LCA (Binary Lifting) */
static const int LOGN = 20; // đủ cho ~10^6
struct LCA {
    int n, LOG;
    vector<int> depth;
    vector<vector<int>> parent;
    vector<vector<int>> adj;
    LCA(int n):n(n){
        LOG = floor(log2(n)) + 1;
        adj.resize(n);
        depth.assign(n,0);
        parent.assign(n, vector<int>(LOG, -1));
    }
    void addEdge(int u,int v){
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    void dfs(int u,int p){
        parent[u][0]=p;
        for(int i=1;i<LOG;i++){
            if(parent[u][i-1] == -1) parent[u][i]=-1; 
            else parent[u][i] = parent[parent[u][i-1]][i-1];
        }
        for(auto &v : adj[u]){
            if(v==p) continue;
            depth[v]=depth[u]+1;
            dfs(v,u);
        }
    }
    int lca(int a,int b){
        if(depth[a]<depth[b]) swap(a,b);
        int diff=depth[a]-depth[b];
        for(int i=0;i<LOG;i++){
            if(diff & (1<<i)){
                a=parent[a][i];
            }
        }
        if(a==b) return a;
        for(int i=LOG-1;i>=0;i--){
            if(parent[a][i]!=parent[b][i]){
                a=parent[a][i];
                b=parent[b][i];
            }
        }
        return parent[a][0];
    }
};

/* 2) Euler Tour (dạng lưu thứ tự vào-ra) */
vector<int> eulerIn, eulerOut, eulerTourArr;
int timerEuler = 0;
void eulerTourDFS(int u,int p, const vector<vector<int>>& adj){
    eulerIn[u] = ++timerEuler;
    eulerTourArr.push_back(u);  // thăm u
    for(auto &v: adj[u]){
        if(v==p) continue;
        eulerTourDFS(v,u,adj);
    }
    eulerOut[u] = timerEuler;
}

/* ===============  XÂU (String)  =============== */

/* 1) Hash / Z / KMP */

/* 1a) Hash (Polynomial Rolling Hash) - ví dụ cơ bản */
struct StringHash {
    long long base, mod;
    vector<long long> prefix, power;
    StringHash(const string &s, long long base=137, long long mod=1000000007) 
        : base(base), mod(mod) {
        int n = (int)s.size();
        prefix.assign(n+1,0);
        power.assign(n+1,1);
        for(int i=0;i<n;i++){
            prefix[i+1] = (prefix[i]*base + s[i])%mod;
            power[i+1] = (power[i]*base)%mod;
        }
    }
    long long getHash(int l,int r){
        // [l, r), r không inclusive
        long long h = prefix[r] - (prefix[l]*power[r-l]%mod);
        if(h<0) h+=mod;
        return h;
    }
};

/* 1b) Z-function */
vector<int> Z_function(const string &s){
    int n=(int)s.size();
    vector<int>z(n,0);
    for(int i=1,l=0,r=0;i<n;i++){
        if(i<=r) z[i] = min(r-i+1, z[i-l]);
        while(i+z[i]<n && s[z[i]]==s[i+z[i]]) z[i]++;
        if(i+z[i]-1>r) l=i,r=i+z[i]-1;
    }
    return z;
}

/* 1c) KMP (pi function) */
vector<int> prefix_function(const string &s){
    int n=(int)s.size();
    vector<int>pi(n,0);
    for(int i=1;i<n;i++){
        int j=pi[i-1];
        while(j>0 && s[i]!=s[j]) j=pi[j-1];
        if(s[i]==s[j]) j++;
        pi[i]=j;
    }
    return pi;
}

/* 2) Manacher (tìm palindrome dài nhất) */
vector<int> manacherOdd(const string &s){
    int n=(int)s.size();
    vector<int>d(n,0);
    int l=0,r=-1;
    for(int i=0;i<n;i++){
        int k=1;
        if(i<=r) k=min(d[l+r-i], r-i+1);
        while(0<=i-k && i+k<n && s[i-k]==s[i+k]) k++;
        d[i]=k--;
        if(i+k>r) l=i-k, r=i+k;
    }
    return d; 
}
/* (d[i] - 1) là bán kính palindrome quanh i, độ dài = 2*d[i] - 1 */

/* 3) Trie cơ bản (chỉ chữ cái a-z) */
struct Trie {
    struct Node {
        bool isEnd;
        Node* nxt[26];
        Node(){
            isEnd=false;
            memset(nxt,0,sizeof(nxt));
        }
    };
    Node* root;
    Trie(){
        root=new Node();
    }
    void insert(const string &s){
        Node* cur=root;
        for(char c: s){
            int idx=c-'a';
            if(!cur->nxt[idx]) cur->nxt[idx]=new Node();
            cur=cur->nxt[idx];
        }
        cur->isEnd=true;
    }
    bool search(const string &s){
        Node* cur=root;
        for(char c:s){
            int idx=c-'a';
            if(!cur->nxt[idx]) return false;
            cur=cur->nxt[idx];
        }
        return cur->isEnd;
    }
};

/* ===============  QUY HOẠCH ĐỘNG  =============== */

/* 1) Dãy con tăng dài nhất (LIS) O(n log n) */
int LIS(vector<int> &a){
    vector<int>temp;
    for(auto &x:a){
        auto it = lower_bound(temp.begin(), temp.end(), x);
        if(it==temp.end()) temp.push_back(x);
        else *it=x;
    }
    return (int)temp.size();
}

/* 2) Xâu con chung dài nhất (LCS) - O(n*m) */
int LCS(const string &s, const string &t){
    int n=(int)s.size(), m=(int)t.size();
    vector<vector<int>> dp(n+1, vector<int>(m+1,0));
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            if(s[i-1]==t[j-1]) dp[i][j]=dp[i-1][j-1]+1;
            else dp[i][j]=max(dp[i-1][j], dp[i][j-1]);
        }
    }
    return dp[n][m];
}

/* 3) DP bitmask - ví dụ bài Matching trong đồ thị n<=20 */
long long dpBitmaskMatching(vector<vector<int>> &w){
    // w[i][j] = cost, n = size
    int n=(int)w.size();
    vector<long long> dp(1<<n,1e15);
    dp[0]=0;
    for(int mask=0; mask<(1<<n); mask++){
        int i=__builtin_popcount(mask);
        for(int j=0;j<n;j++){
            if(!(mask&(1<<j))){
                dp[mask|(1<<j)] = min(dp[mask|(1<<j)], dp[mask] + w[i][j]);
            }
        }
    }
    return dp[(1<<n)-1];
}

/* 4) Knapsack 1 (cơ bản) O(n*W) */
long long knapsack(const vector<int> &wt, const vector<int> &val, int W){
    // wt[i], val[i], W = sức chứa
    // trả về giá trị lớn nhất
    int n=(int)wt.size();
    vector<long long> dp(W+1,0LL);
    for(int i=0;i<n;i++){
        for(int cap=W;cap>=wt[i];cap--){
            dp[cap] = max(dp[cap], dp[cap-wt[i]] + val[i]);
        }
    }
    return dp[W];
}

/* 5) Quy hoạch động chữ số (Digit DP) - khung cơ bản 
   Thường ta cài 1 hàm đệ quy tính f(pos, sum, tight, digits...) */
long long digitDP_util(int idx, long long sum, bool tight, const vector<int>& digits, vector<vector<long long>> &dp){
    // Ví dụ: đếm số cách sao cho sum <= 9..., tuỳ bài
    // Đây chỉ là khung sườn, tuỳ đề mà ta mở rộng state.
    if(idx == (int)digits.size()) {
        return (sum == 0); // hay 1, tuỳ yêu cầu
    }
    if(!tight && dp[idx][sum]!=-1) return dp[idx][sum];
    int limit = (tight ? digits[idx] : 9);
    long long res=0;
    for(int dig=0; dig<=limit; dig++){
        res += digitDP_util(idx+1, (sum+dig), tight && (dig==limit), digits, dp);
    }
    if(!tight) dp[idx][sum]=res;
    return res;
}
long long digitDP(long long x){
    // Chuyển x thành mảng chữ số
    if(x<0) return 0;
    vector<int> digits;
    while(x>0){digits.push_back(x%10); x/=10;}
    reverse(digits.begin(), digits.end());
    vector<vector<long long>> dp(digits.size()+1, vector<long long>(100, -1));
    // tuỳ yêu cầu state sum = 0..99 (ví dụ)
    return digitDP_util(0, 0, true, digits, dp);
}

/* 6) DP trên cây (ví dụ đếm số con độc lập, v.v) 
   - tuỳ bài, chúng ta cài đặt hàm dfs(u, p) rồi dp[u] 
*/

/* 7) DP trên DAG (Longest Path) */
vector<long long> DAGLongestPath(const vector<vector<int>>& adj, const vector<int>& topoOrder){
    int n=(int)adj.size();
    vector<long long> dist(n,0LL);
    // Giả sử đã có topoOrder sẵn
    for(auto &u: topoOrder){
        for(auto &v: adj[u]){
            dist[v] = max(dist[v], dist[u]+1);
        }
    }
    return dist;
}

/* 8) DP nhân ma trận (Matrix Chain Multiplication) - O(n^3) */
long long matrixChain(vector<int>& dims){
    // dims là vector chiều, length = n+1 nếu có n ma trận
    int n=(int)dims.size()-1; 
    vector<vector<long long>> dp(n, vector<long long>(n,0));
    for(int len=2; len<=n; len++){
        for(int i=0; i<=n-len; i++){
            int j=i+len-1;
            dp[i][j]=LLONG_MAX;
            for(int k=i; k<j; k++){
                long long cost = dp[i][k]+dp[k+1][j]+(long long)dims[i]*dims[k+1]*dims[j+1];
                dp[i][j]=min(dp[i][j], cost);
            }
        }
    }
    return dp[0][n-1];
}

/* 9) DP bao lồi (Convex Hull Trick) 
   Chỉ mang tính gợi ý, tuỳ công thức f(x)=mx+b. Thêm/sửa tuỳ bài.
*/
struct ConvexHullTrick {
    struct Line {
        long long m,b;
        long double intersect(Line o) {
            // Giao của mx+b = o.m x + o.b => x = (o.b - b)/(m - o.m)
            return (long double)(o.b - b)/(m - o.m);
        }
    };
    deque<Line> hull;
    void addLine(long long m, long long b){
        Line l={m,b};
        // Xoá khỏi đuôi nếu ko cần
        while(hull.size()>=2){
            auto &l1=hull[hull.size()-2];
            auto &l2=hull[hull.size()-1];
            long double x1=l1.intersect(l2);
            long double x2=l1.intersect(l);
            if(x2 <= x1) hull.pop_back();
            else break;
        }
        hull.push_back(l);
    }
    long long getVal(Line &l, long long x){
        return l.m*x + l.b;
    }
    long long query(long long x){
        // Tìm line tốt nhất, ở đây x tăng
        while(hull.size()>=2){
            auto &l1=hull[0];
            auto &l2=hull[1];
            if(getVal(l1,x)>getVal(l2,x)) hull.pop_front();
            else break;
        }
        return getVal(hull[0], x);
    }
};

/* 10) DP theo thứ tự từ điển (ví dụ “Số hiệu tổ hợp”) -> có thể tham khảo C(n,k) + logic so sánh 
   Tùy đề bài, không có code chung */

/* 11) DP SOS (Sum over subset) - Codeforces 165E khung 
   dp[mask] = sum dp[submask]? 
   Chúng ta có thể cài đặt 1 loop FWT (Fast Walsh Transform) hoặc gói gọn logic */
void SOS(vector<long long> &f){
    // f[mask] ban đầu
    int n=(int)f.size();
    int LOG = 0;
    while((1<<LOG)<n) LOG++;
    for(int i=0;i<LOG;i++){
        for(int mask=0; mask<n; mask++){
            if(mask & (1<<i)){
                f[mask]+=f[mask^(1<<i)];
            }
        }
    }
}

/* ===============  HÌNH HỌC  =============== */

/* 1) Bao lồi (Convex Hull) - Monotone Chain */
struct pt {
    long long x, y;
};
long long cross(pt A, pt B, pt C){
    return (B.x-A.x)*(C.y-A.y) - (B.y-A.y)*(C.x-A.x);
}
vector<pt> convexHull(vector<pt> &pts){
    sort(pts.begin(), pts.end(), [](pt &a,pt &b){
        if(a.x!=b.x) return a.x<b.x;
        return a.y<b.y;
    });
    vector<pt> hull;
    // build lower hull
    for(auto &p: pts){
        while(hull.size()>=2 && cross(hull[hull.size()-2], hull[hull.size()-1], p)<=0){
            hull.pop_back();
        }
        hull.push_back(p);
    }
    // build upper hull
    for(int i=(int)pts.size()-2, t=hull.size()+1; i>=0; i--){
        while((int)hull.size()>=t && cross(hull[hull.size()-2], hull[hull.size()-1], pts[i])<=0){
            hull.pop_back();
        }
        hull.push_back(pts[i]);
    }
    hull.pop_back();
    return hull;
}

/* 2) Hướng của 3 điểm (CCW) */
int ccw(pt A, pt B, pt C){
    long long cr = cross(A,B,C);
    if(cr>0) return 1;   // A->B->C quay trái
    if(cr<0) return -1;  // quay phải
    return 0;            // thẳng hàng
}

/* 3) Kiểm tra 2 đoạn thẳng giao nhau */
bool intersect1d(long long l1, long long r1, long long l2, long long r2){
    if(l1>r1) swap(l1,r1);
    if(l2>r2) swap(l2,r2);
    return max(l1,l2)<=min(r1,r2);
}
bool segmentIntersect(pt A, pt B, pt C, pt D){
    // Kiểm tra A-B cắt C-D
    return intersect1d(A.x,B.x,C.x,D.x) && 
           intersect1d(A.y,B.y,C.y,D.y) &&
           ccw(A,B,C)*ccw(A,B,D)<=0 &&
           ccw(C,D,A)*ccw(C,D,B)<=0;
}

/* 4) Tính diện tích đa giác (polygon area) */
long long polygonArea(const vector<pt> &poly){
    long long area=0;
    int n=(int)poly.size();
    for(int i=0;i<n;i++){
        pt A=poly[i], B=poly[(i+1)%n];
        area+=A.x*B.y - A.y*B.x;
    }
    return llabs(area);
}

/* 5) Kiểm tra điểm trong đa giác (point in polygon) - ray casting cơ bản */
bool pointInPolygon(pt P, const vector<pt> &poly){
    int cnt=0, n=(int)poly.size();
    for(int i=0;i<n;i++){
        pt A=poly[i], B=poly[(i+1)%n];
        if(A.y==B.y) continue; // đường ngang
        if(P.y<min(A.y,B.y) || P.y>=max(A.y,B.y)) continue;
        double xint = (double)(P.y-A.y)*(B.x-A.x)/(B.y-A.y)+A.x;
        if(xint>P.x) cnt^=1;
    }
    return cnt; // lẻ => trong, chẵn => ngoài
}

/* ===============  KHÁC  =============== */

/* 1) Quay lui (Backtracking) - ví dụ sinh xâu nhị phân độ dài n */
void backtrackBinaryString(int idx, int n, string &temp){
    if(idx==n){
        // xử lý temp
        // cout << temp << "\n";
        return;
    }
    for(char c='0'; c<='1'; c++){
        temp[idx]=c;
        backtrackBinaryString(idx+1,n,temp);
    }
}

/* 2) Rời rạc hoá (Coordinate Compression) */
vector<int> compress(vector<int> arr){
    vector<int> vals=arr;
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    // map arr[i] -> vị trí trong vals
    for(auto &x: arr){
        x = (int)(lower_bound(vals.begin(), vals.end(), x) - vals.begin());
    }
    return arr; // đã bị thay đổi
}

/* 3) Chia căn (sqrt decomposition) - D-query kiểu Mo's algorithm hoặc sqrt cho Segment
   Tuỳ bài, dưới đây là khung sườn Mo's algorithm (offline). 
*/
struct Query {
    int L,R, idx;
};
static int BLOCK;
bool cmpMo(Query &a, Query &b){
    if(a.L/BLOCK != b.L/BLOCK) return a.L < b.L;
    return ((a.L/BLOCK)&1) ? (a.R < b.R) : (a.R > b.R);
}
vector<long long> moAlgorithm(vector<int> &arr, vector<Query> &queries){
    BLOCK = (int)sqrt(arr.size());
    sort(queries.begin(), queries.end(), cmpMo);
    vector<long long> ans(queries.size());
    int curL=0, curR=-1;
    long long curAns=0;
    auto add = [&](int pos){
        // thêm arr[pos] vào tập
        curAns += arr[pos]; // ví dụ
    };
    auto remove = [&](int pos){
        // xoá arr[pos] khỏi tập
        curAns -= arr[pos]; // ví dụ
    };
    for(auto &q: queries){
        while(curR<q.R) add(++curR);
        while(curR>q.R) remove(curR--);
        while(curL<q.L) remove(curL++);
        while(curL>q.L) add(--curL);
        ans[q.idx]=curAns;
    }
    return ans;
}

/* ===================== HẾT ===================== */
