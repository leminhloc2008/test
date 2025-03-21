/**
 * Comprehensive Data Structures Template
 * Includes implementations of:
 * - Segment Trees (Classic, Lazy Propagation, Persistent, Sparse)
 * - Fenwick Trees (Binary Indexed Tree, 2D BIT)
 * - Treap (Implicit, Persistent)
 * - Sparse Table
 * - Disjoint Set Union (DSU)
 * - Trie
 * - Suffix Array and LCP
 * - Heavy-Light Decomposition
 * - Centroid Decomposition
 * - Splay Tree
 * - Link-Cut Tree
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>
#include <random>
#include <climits>
#include <cmath>
#include <string>
#include <chrono>
using namespace std;

typedef long long ll;
typedef pair<int, int> pii;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

//------------------
// SEGMENT TREE
//------------------

// Classic Segment Tree (Min/Max/Sum queries)
class SegmentTree {
private:
    vector<int> tree;
    int n;
    
    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
            return;
        }
        int mid = (start + end) / 2;
        build(arr, 2 * node, start, mid);
        build(arr, 2 * node + 1, mid + 1, end);
        tree[node] = tree[2 * node] + tree[2 * node + 1]; // Change operation as needed (min, max, sum)
    }
    
    void update(int node, int start, int end, int idx, int val) {
        if (start == end) {
            tree[node] = val;
            return;
        }
        int mid = (start + end) / 2;
        if (idx <= mid)
            update(2 * node, start, mid, idx, val);
        else
            update(2 * node + 1, mid + 1, end, idx, val);
        tree[node] = tree[2 * node] + tree[2 * node + 1]; // Change operation as needed
    }
    
    int query(int node, int start, int end, int l, int r) {
        if (r < start || end < l)
            return 0; // Change return value for different operations (e.g., INT_MAX for min)
        if (l <= start && end <= r)
            return tree[node];
        int mid = (start + end) / 2;
        int p1 = query(2 * node, start, mid, l, r);
        int p2 = query(2 * node + 1, mid + 1, end, l, r);
        return p1 + p2; // Change operation as needed
    }
    
public:
    SegmentTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        build(arr, 1, 0, n - 1);
    }
    
    void update(int idx, int val) {
        update(1, 0, n - 1, idx, val);
    }
    
    int query(int l, int r) {
        return query(1, 0, n - 1, l, r);
    }
};

// Lazy Propagation Segment Tree (Range Updates)
class LazySegmentTree {
private:
    vector<int> tree, lazy;
    int n;
    
    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
            return;
        }
        int mid = (start + end) / 2;
        build(arr, 2 * node, start, mid);
        build(arr, 2 * node + 1, mid + 1, end);
        tree[node] = tree[2 * node] + tree[2 * node + 1];
    }
    
    void pushDown(int node, int start, int end) {
        if (lazy[node] != 0) {
            tree[node] += (end - start + 1) * lazy[node]; // For sum
            if (start != end) {
                lazy[2 * node] += lazy[node];
                lazy[2 * node + 1] += lazy[node];
            }
            lazy[node] = 0;
        }
    }
    
    void updateRange(int node, int start, int end, int l, int r, int val) {
        pushDown(node, start, end);
        if (start > end || start > r || end < l)
            return;
        if (start >= l && end <= r) {
            tree[node] += (end - start + 1) * val; // For sum
            if (start != end) {
                lazy[2 * node] += val;
                lazy[2 * node + 1] += val;
            }
            return;
        }
        int mid = (start + end) / 2;
        updateRange(2 * node, start, mid, l, r, val);
        updateRange(2 * node + 1, mid + 1, end, l, r, val);
        tree[node] = tree[2 * node] + tree[2 * node + 1];
    }
    
    int queryRange(int node, int start, int end, int l, int r) {
        if (start > end || start > r || end < l)
            return 0; // For sum
        pushDown(node, start, end);
        if (start >= l && end <= r)
            return tree[node];
        int mid = (start + end) / 2;
        int p1 = queryRange(2 * node, start, mid, l, r);
        int p2 = queryRange(2 * node + 1, mid + 1, end, l, r);
        return p1 + p2;
    }
    
public:
    LazySegmentTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        lazy.resize(4 * n, 0);
        build(arr, 1, 0, n - 1);
    }
    
    void updateRange(int l, int r, int val) {
        updateRange(1, 0, n - 1, l, r, val);
    }
    
    int queryRange(int l, int r) {
        return queryRange(1, 0, n - 1, l, r);
    }
};

// Persistent Segment Tree
class PersistentSegmentTree {
private:
    struct Node {
        int val;
        Node *left, *right;
        Node(int v = 0, Node* l = nullptr, Node* r = nullptr) : val(v), left(l), right(r) {}
    };
    
    vector<Node*> roots;
    int n;
    
    Node* build(const vector<int>& arr, int start, int end) {
        if (start == end)
            return new Node(arr[start]);
        int mid = (start + end) / 2;
        Node* left = build(arr, start, mid);
        Node* right = build(arr, mid + 1, end);
        return new Node(left->val + right->val, left, right);
    }
    
    Node* update(Node* node, int start, int end, int idx, int val) {
        if (start == end)
            return new Node(val);
        int mid = (start + end) / 2;
        Node* left = node->left;
        Node* right = node->right;
        if (idx <= mid)
            left = update(node->left, start, mid, idx, val);
        else
            right = update(node->right, mid + 1, end, idx, val);
        return new Node(left->val + right->val, left, right);
    }
    
    int query(Node* node, int start, int end, int l, int r) {
        if (r < start || end < l)
            return 0;
        if (l <= start && end <= r)
            return node->val;
        int mid = (start + end) / 2;
        int p1 = query(node->left, start, mid, l, r);
        int p2 = query(node->right, mid + 1, end, l, r);
        return p1 + p2;
    }
    
public:
    PersistentSegmentTree(const vector<int>& arr) {
        n = arr.size();
        roots.push_back(build(arr, 0, n - 1));
    }
    
    void update(int version, int idx, int val) {
        roots.push_back(update(roots[version], 0, n - 1, idx, val));
    }
    
    int query(int version, int l, int r) {
        return query(roots[version], 0, n - 1, l, r);
    }
    
    int getLatestVersion() {
        return roots.size() - 1;
    }
};

// Sparse Segment Tree (for large indices up to 1e9)
class SparseSegmentTree {
private:
    struct Node {
        int val;
        Node *left, *right;
        Node(int v = 0) : val(v), left(nullptr), right(nullptr) {}
    };
    
    Node* root;
    int minIdx, maxIdx;
    
    void update(Node* &node, int start, int end, int idx, int val) {
        if (!node)
            node = new Node();
        if (start == end) {
            node->val = val;
            return;
        }
        int mid = start + (end - start) / 2;
        if (idx <= mid)
            update(node->left, start, mid, idx, val);
        else
            update(node->right, mid + 1, end, idx, val);
        node->val = (node->left ? node->left->val : 0) + (node->right ? node->right->val : 0);
    }
    
    int query(Node* node, int start, int end, int l, int r) {
        if (!node || r < start || end < l)
            return 0;
        if (l <= start && end <= r)
            return node->val;
        int mid = start + (end - start) / 2;
        int p1 = query(node->left, start, mid, l, r);
        int p2 = query(node->right, mid + 1, end, l, r);
        return p1 + p2;
    }
    
public:
    SparseSegmentTree(int min_idx, int max_idx) : minIdx(min_idx), maxIdx(max_idx), root(nullptr) {}
    
    void update(int idx, int val) {
        update(root, minIdx, maxIdx, idx, val);
    }
    
    int query(int l, int r) {
        return query(root, minIdx, maxIdx, l, r);
    }
};

// Dynamic Segment Tree (with allocation on insertion, for large sparse ranges)
class DynamicSegmentTree {
private:
    struct Node {
        int sum;
        Node *left, *right;
        Node() : sum(0), left(nullptr), right(nullptr) {}
    };
    
    Node* root;
    int minRange, maxRange;
    
    void update(Node* &node, int start, int end, int idx, int val) {
        if (!node)
            node = new Node();
        if (start == end) {
            node->sum += val;
            return;
        }
        int mid = start + (end - start) / 2;
        if (idx <= mid)
            update(node->left, start, mid, idx, val);
        else
            update(node->right, mid + 1, end, idx, val);
        node->sum = (node->left ? node->left->sum : 0) + (node->right ? node->right->sum : 0);
    }
    
    int query(Node* node, int start, int end, int l, int r) {
        if (!node || r < start || end < l)
            return 0;
        if (l <= start && end <= r)
            return node->sum;
        int mid = start + (end - start) / 2;
        int leftSum = query(node->left, start, mid, l, r);
        int rightSum = query(node->right, mid + 1, end, l, r);
        return leftSum + rightSum;
    }
    
public:
    // For really large ranges, e.g., minRange=-1e9, maxRange=1e9
    DynamicSegmentTree(int minR, int maxR) : minRange(minR), maxRange(maxR), root(nullptr) {}
    
    void update(int idx, int val) {
        update(root, minRange, maxRange, idx, val);
    }
    
    int query(int l, int r) {
        return query(root, minRange, maxRange, l, r);
    }
};

// Merge Sort Tree (for counting elements in range)
class MergeSortTree {
private:
    vector<vector<int>> tree;
    int n;
    
    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node].push_back(arr[start]);
            return;
        }
        int mid = (start + end) / 2;
        build(arr, 2 * node, start, mid);
        build(arr, 2 * node + 1, mid + 1, end);
        
        // Merge the sorted lists
        merge(tree[2 * node].begin(), tree[2 * node].end(),
              tree[2 * node + 1].begin(), tree[2 * node + 1].end(),
              back_inserter(tree[node]));
    }
    
    // Count elements less than or equal to val in range [l,r]
    int countLessEqual(int node, int start, int end, int l, int r, int val) {
        if (r < start || end < l)
            return 0;
        if (l <= start && end <= r)
            return upper_bound(tree[node].begin(), tree[node].end(), val) - tree[node].begin();
        int mid = (start + end) / 2;
        int p1 = countLessEqual(2 * node, start, mid, l, r, val);
        int p2 = countLessEqual(2 * node + 1, mid + 1, end, l, r, val);
        return p1 + p2;
    }
    
public:
    MergeSortTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        build(arr, 1, 0, n - 1);
    }
    
    // Count elements in range [l,r] less than or equal to val
    int countLessEqual(int l, int r, int val) {
        return countLessEqual(1, 0, n - 1, l, r, val);
    }
    
    // Count elements in range [l,r] within range [lo,hi]
    int countRange(int l, int r, int lo, int hi) {
        return countLessEqual(l, r, hi) - countLessEqual(l, r, lo - 1);
    }
};

//------------------
// FENWICK TREE (BIT)
//------------------

// 1D Fenwick Tree (Binary Indexed Tree)
class FenwickTree {
private:
    vector<int> bit;
    int n;
    
    int lsb(int i) {
        return i & -i;
    }
    
public:
    FenwickTree(int size) : n(size + 1) {
        bit.resize(n, 0);
    }
    
    FenwickTree(const vector<int>& arr) : n(arr.size() + 1) {
        bit.resize(n, 0);
        for (int i = 0; i < arr.size(); i++)
            update(i, arr[i]);
    }
    
    // Add val to index i
    void update(int i, int val) {
        i++; // 1-indexed internally
        while (i < n) {
            bit[i] += val;
            i += lsb(i);
        }
    }
    
    // Get sum of elements in range [0,i]
    int prefixSum(int i) {
        i++; // 1-indexed internally
        int sum = 0;
        while (i > 0) {
            sum += bit[i];
            i -= lsb(i);
        }
        return sum;
    }
    
    // Get sum of elements in range [l,r]
    int rangeSum(int l, int r) {
        return prefixSum(r) - prefixSum(l - 1);
    }
};

// 2D Fenwick Tree
class FenwickTree2D {
private:
    vector<vector<int>> bit;
    int n, m;
    
    int lsb(int i) {
        return i & -i;
    }
    
public:
    FenwickTree2D(int rows, int cols) : n(rows + 1), m(cols + 1) {
        bit.resize(n, vector<int>(m, 0));
    }
    
    // Add val at position (x,y)
    void update(int x, int y, int val) {
        x++; y++; // 1-indexed internally
        for (int i = x; i < n; i += lsb(i))
            for (int j = y; j < m; j += lsb(j))
                bit[i][j] += val;
    }
    
    // Get sum in rectangle [(0,0), (x,y)]
    int prefixSum(int x, int y) {
        x++; y++; // 1-indexed internally
        int sum = 0;
        for (int i = x; i > 0; i -= lsb(i))
            for (int j = y; j > 0; j -= lsb(j))
                sum += bit[i][j];
        return sum;
    }
    
    // Get sum in rectangle [(x1,y1), (x2,y2)]
    int rangeSum(int x1, int y1, int x2, int y2) {
        return prefixSum(x2, y2) - prefixSum(x1 - 1, y2) - prefixSum(x2, y1 - 1) + prefixSum(x1 - 1, y1 - 1);
    }
};

// Range Update Point Query BIT
class RangeUpdateBIT {
private:
    vector<int> bit;
    int n;
    
    int lsb(int i) {
        return i & -i;
    }
    
public:
    RangeUpdateBIT(int size) : n(size + 1) {
        bit.resize(n, 0);
    }
    
    // Add val to all elements in range [l,r]
    void updateRange(int l, int r, int val) {
        update(l, val);
        update(r + 1, -val);
    }
    
    // Helper for range update
    void update(int i, int val) {
        i++; // 1-indexed internally
        while (i < n) {
            bit[i] += val;
            i += lsb(i);
        }
    }
    
    // Get value at index i
    int pointQuery(int i) {
        i++; // 1-indexed internally
        int sum = 0;
        while (i > 0) {
            sum += bit[i];
            i -= lsb(i);
        }
        return sum;
    }
};

// Range Update Range Query BIT
class RangeUpdateRangeQueryBIT {
private:
    vector<int> bit1, bit2;
    int n;
    
    int lsb(int i) {
        return i & -i;
    }
    
    void updateBIT(vector<int>& bit, int i, int val) {
        while (i < n) {
            bit[i] += val;
            i += lsb(i);
        }
    }
    
    int queryBIT(vector<int>& bit, int i) {
        int sum = 0;
        while (i > 0) {
            sum += bit[i];
            i -= lsb(i);
        }
        return sum;
    }
    
public:
    RangeUpdateRangeQueryBIT(int size) : n(size + 1) {
        bit1.resize(n, 0);
        bit2.resize(n, 0);
    }
    
    // Add val to all elements in range [l,r]
    void updateRange(int l, int r, int val) {
        l++; r++; // 1-indexed internally
        
        // For BIT1: a[i] += val
        updateBIT(bit1, l, val);
        updateBIT(bit1, r + 1, -val);
        
        // For BIT2: a[i] * i
        updateBIT(bit2, l, val * (l - 1));
        updateBIT(bit2, r + 1, -val * r);
    }
    
    // Get sum of elements in range [1,i]
    int prefixSum(int i) {
        i++; // 1-indexed internally
        return queryBIT(bit1, i) * i - queryBIT(bit2, i);
    }
    
    // Get sum of elements in range [l,r]
    int rangeSum(int l, int r) {
        return prefixSum(r) - prefixSum(l - 1);
    }
};

//------------------
// SPARSE TABLE
//------------------

// Sparse Table for RMQ (Range Minimum Query)
class SparseTable {
private:
    vector<vector<int>> st;
    vector<int> log_table;
    int n;
    
public:
    SparseTable(const vector<int>& arr) {
        n = arr.size();
        int max_log = 32 - __builtin_clz(n);
        
        // Precompute log values
        log_table.resize(n + 1);
        log_table[1] = 0;
        for (int i = 2; i <= n; i++)
            log_table[i] = log_table[i/2] + 1;
        
        // Build sparse table
        st.resize(n, vector<int>(max_log + 1));
        
        // Initialize for lengths of 2^0 = 1
        for (int i = 0; i < n; i++)
            st[i][0] = arr[i];
        
        // Build table for larger ranges
        for (int j = 1; j <= max_log; j++)
            for (int i = 0; i + (1 << j) <= n; i++)
                st[i][j] = min(st[i][j-1], st[i + (1 << (j-1))][j-1]);
    }
    
    // Query for minimum value in range [l,r]
    int queryMin(int l, int r) {
        int len = r - l + 1;
        int j = log_table[len];
        return min(st[l][j], st[r - (1 << j) + 1][j]);
    }
};

// Sparse Table for GCD (Greatest Common Divisor)
class SparseTableGCD {
private:
    vector<vector<int>> st;
    vector<int> log_table;
    int n;
    
    int gcd(int a, int b) {
        while (b) {
            a %= b;
            swap(a, b);
        }
        return a;
    }
    
public:
    SparseTableGCD(const vector<int>& arr) {
        n = arr.size();
        int max_log = 32 - __builtin_clz(n);
        
        // Precompute log values
        log_table.resize(n + 1);
        log_table[1] = 0;
        for (int i = 2; i <= n; i++)
            log_table[i] = log_table[i/2] + 1;
        
        // Build sparse table
        st.resize(n, vector<int>(max_log + 1));
        
        // Initialize for lengths of 2^0 = 1
        for (int i = 0; i < n; i++)
            st[i][0] = arr[i];
        
        // Build table for larger ranges
        for (int j = 1; j <= max_log; j++)
            for (int i = 0; i + (1 << j) <= n; i++)
                st[i][j] = gcd(st[i][j-1], st[i + (1 << (j-1))][j-1]);
    }
    
    // Query for GCD in range [l,r]
    int queryGCD(int l, int r) {
        int len = r - l + 1;
        int j = log_table[len];
        return gcd(st[l][j], st[r - (1 << j) + 1][j]);
    }
};

//------------------
// TREAP
//------------------

// Basic Treap (Randomized BST)
class Treap {
private:
    struct Node {
        int key, priority;
        Node *left, *right;
        
        Node(int k) : key(k), priority(rng()), left(nullptr), right(nullptr) {}
    };
    
    Node* root;
    
    // Split treap into nodes with keys <= key and > key
    pair<Node*, Node*> split(Node* t, int key) {
        if (!t)
            return {nullptr, nullptr};
        
        if (key < t->key) {
            auto [l, r] = split(t->left, key);
            t->left = r;
            return {l, t};
        } else {
            auto [l, r] = split(t->right, key);
            t->right = l;
            return {t, r};
        }
    }
    
    // Merge two treaps (all keys in l <= all keys in r)
    Node* merge(Node* l, Node* r) {
        if (!l || !r)
            return l ? l : r;
        
        if (l->priority > r->priority) {
            l->right = merge(l->right, r);
            return l;
        } else {
            r->left = merge(l, r->left);
            return r;
        }
    }
    
    // Insert key into treap
    Node* insert(Node* t, int key) {
        if (!t)
            return new Node(key);
        
        if (t->priority < rng()) {
            Node* newNode = new Node(key);
            auto [l, r] = split(t, key);
            newNode->left = l;
            newNode->right = r;
            return newNode;
        }
        
        if (key < t->key)
            t->left = insert(t->left, key);
        else
            t->right = insert(t->right, key);
        return t;
    }
    
    // Erase key from treap
    Node* erase(Node* t, int key) {
        if (!t)
            return nullptr;
        
        if (t->key == key)
            return merge(t->left, t->right);
        
        if (key < t->key)
            t->left = erase(t->left, key);
        else
            t->right = erase(t->right, key);
        return t;
    }
    
    // Check if key exists
    bool find(Node* t, int key) {
        if (!t)
            return false;
        if (t->key == key)
            return true;
        if (key < t->key)
            return find(t->left, key);
        else
            return find(t->right, key);
    }
    
public:
    Treap() : root(nullptr) {}
    
    void insert(int key) {
        root = insert(root, key);
    }
    
    void erase(int key) {
        root = erase(root, key);
    }
    
    bool find(int key) {
        return find(root, key);
    }
};

// Implicit Treap (for array operations)
class ImplicitTreap {
private:
    struct Node {
        int priority, size;
        int value; // Value stored at this node
        int sum;   // Sum of subtree (can be changed to other operations)
        Node *left, *right;
        
        Node(int v) : priority(rng()), size(1), value(v), sum(v), left(nullptr), right(nullptr) {}
    };
    
    Node* root;
    
    // Update size and sum of node
    void update(Node* t) {
        if (!t) return;
        t->size = 1 + (t->left ? t->left->size : 0) + (t->right ? t->right->size : 0);
        t->sum = t->value + (t->left ? t->left->sum : 0) + (t->right ? t->right->sum : 0);
    }
    
    // Split treap into [0, k) and [k, size)
    pair<Node*, Node*> split(Node* t, int k) {
        if (!t) return {nullptr, nullptr};
        
        int leftSize = t->left ? t->left->size : 0;
        if (leftSize >= k) {
            auto [l, r] = split(t->left, k);
            t->left = r;
            update(t);
            return {l, t};
        } else {
            auto [l, r] = split(t->right, k - leftSize - 1);
            t->right = l;
            update(t);
            return {t, r};
        }
    }
    
    // Merge two treaps
    Node* merge(Node* l, Node* r) {
        if (!l || !r) return l ? l : r;
        
        if (l->priority > r->priority) {
            l->right = merge(l->right, r);
            update(l);
            return l;
        } else {
            r->left = merge(l, r->left);
            update(r);
            return r;
        }
    }
    
    // Insert value at position k
    Node* insert(Node* t, int k, int value) {
        auto [l, r] = split(t, k);
        Node* newNode = new Node(value);
        return merge(merge(l, newNode), r);
    }
    
    // Erase position k
    Node* erase(Node* t, int k) {
        auto [l, mid_r] = split(t, k);
        auto [m, r] = split(mid_r, 1);
        return merge(l, r);
    }
    
    // Get sum in range [l, r]
    int rangeSum(Node* t, int l, int r) {
        auto [left, mid_right] = split(t, l);
        auto [mid, right] = split(mid_right, r - l + 1);
        int result = mid ? mid->sum : 0;
        t = merge(left, merge(mid, right));
        return result;
    }
    
public:
    ImplicitTreap() : root(nullptr) {}
    
    void insert(int k, int value) {
        root = insert(root, k, value);
    }
    
    void erase(int k) {
        root = erase(root, k);
    }
    
    int rangeSum(int l, int r) {
        return rangeSum(root, l, r);
    }
    
    int size() {
        return root ? root->size : 0;
    }
};

//------------------
// DISJOINT SET UNION (DSU)
//------------------

class DisjointSetUnion {
private:
    vector<int> parent, rank;
    
public:
    DisjointSetUnion(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++)
            parent[i] = i;
    }
    
    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }
    
    void unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX == rootY)
            return;
        
        if (rank[rootX] < rank[rootY])
            parent[rootX] = rootY;
        else if (rank[rootX] > rank[rootY])
            parent[rootY] = rootX;
        else {
            parent[rootY] = rootX;
            rank[rootX]++;
        }
    }
    
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};

// DSU with Path Compression and Union by Size
class DSUWithSize {
private:
    vector<int> parent, size;
    int components;
    
public:
    DSUWithSize(int n) : components(n) {
        parent.resize(n);
        size.resize(n, 1);
        for (int i = 0; i < n; i++)
            parent[i] = i;
    }
    
    int find(int x) {
        return parent[x] == x ? x : (parent[x] = find(parent[x]));
    }
    
    bool unite(int x, int y) {
        x = find(x);
        y = find(y);
        
        if (x == y)
            return false;
        
        if (size[x] < size[y])
            swap(x, y);
            
        parent[y] = x;
        size[x] += size[y];
        components--;
        return true;
    }
    
    int getSize(int x) {
        return size[find(x)];
    }
    
    int getComponentCount() {
        return components;
    }
};

//------------------
// TRIE
//------------------

// Basic Trie (for strings)
class Trie {
private:
    struct Node {
        Node* children[26];
        bool isEnd;
        
        Node() : isEnd(false) {
            for (int i = 0; i < 26; i++)
                children[i] = nullptr;
        }
    };
    
    Node* root;
    
public:
    Trie() : root(new Node()) {}
    
    void insert(string word) {
        Node* node = root;
        for (char c : word) {
            int index = c - 'a';
            if (!node->children[index])
                node->children[index] = new Node();
            node = node->children[index];
        }
        node->isEnd = true;
    }
    
    bool search(string word) {
        Node* node = root;
        for (char c : word) {
            int index = c - 'a';
            if (!node->children[index])
                return false;
            node = node->children[index];
        }
        return node->isEnd;
    }
    
    bool startsWith(string prefix) {
        Node* node = root;
        for (char c : prefix) {
            int index = c - 'a';
            if (!node->children[index])
                return false;
            node = node->children[index];
        }
        return true;
    }
};

// Compressed Trie (for strings with common prefixes)
class CompressedTrie {
private:
    struct Node {
        unordered_map<string, Node*> children;
        bool isEnd;
        
        Node() : isEnd(false) {}
    };
    
    Node* root;
    
    // Find longest common prefix
    string longestCommonPrefix(const string& s1, const string& s2) {
        int minLen = min(s1.length(), s2.length());
        for (int i = 0; i < minLen; i++)
            if (s1[i] != s2[i])
                return s1.substr(0, i);
        return s1.substr(0, minLen);
    }
    
public:
    CompressedTrie() : root(new Node()) {}
    
    void insert(string word) {
        Node* node = root;
        while (!word.empty()) {
            bool found = false;
            for (auto& [edge, child] : node->children) {
                string lcp = longestCommonPrefix(edge, word);
                if (!lcp.empty()) {
                    if (lcp.length() == edge.length()) {
                        // Edge is completely matched
                        word = word.substr(lcp.length());
                        node = child;
                    } else {
                        // Edge is partially matched, split the edge
                        Node* mid = new Node();
                        mid->children[edge.substr(lcp.length())] = child;
                        node->children[lcp] = mid;
                        node->children.erase(edge);
                        word = word.substr(lcp.length());
                        node = mid;
                    }
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                // No matching edge, create new
                Node* newNode = new Node();
                node->children[word] = newNode;
                node = newNode;
                word = "";
            }
        }
        node->isEnd = true;
    }
    
    bool search(string word) {
        Node* node = root;
        while (!word.empty()) {
            bool found = false;
            for (auto& [edge, child] : node->children) {
                if (word.length() >= edge.length() && word.substr(0, edge.length()) == edge) {
                    word = word.substr(edge.length());
                    node = child;
                    found = true;
                    break;
                }
            }
            if (!found)
                return false;
        }
        return node->isEnd;
    }
};

//------------------
// SUFFIX ARRAY AND LCP
//------------------

// Suffix Array and LCP (Longest Common Prefix)
class SuffixArray {
private:
    string s;
    vector<int> sa;      // suffix array
    vector<int> rank;    // rank of each suffix
    vector<int> lcp;     // lcp[i] = LCP(sa[i], sa[i+1])
    
public:
    SuffixArray(string str) : s(str) {
        int n = s.length();
        sa.resize(n);
        buildSuffixArray(n);
        buildLCP(n);
    }
    
    // Build suffix array using counting sort (O(n log n))
    void buildSuffixArray(int n) {
        vector<int> count(max(256, n), 0);
        rank.resize(n);
        
        // Initialize rank array with character values
        for (int i = 0; i < n; i++)
            rank[i] = s[i];
        
        vector<int> tempSA(n);
        vector<int> tempRank(n);
        
        for (int gap = 1; gap < n; gap *= 2) {
            // Sort by second component
            fill(count.begin(), count.end(), 0);
            for (int i = 0; i < n; i++)
                count[(i + gap < n) ? rank[i + gap] : 0]++;
                
            for (int i = 1; i < count.size(); i++)
                count[i] += count[i - 1];
                
            for (int i = n - 1; i >= 0; i--)
                tempSA[--count[(i + gap < n) ? rank[i + gap] : 0]] = i;
            
            // Sort by first component
            fill(count.begin(), count.end(), 0);
            for (int i = 0; i < n; i++)
                count[rank[tempSA[i]]]++;
                
            for (int i = 1; i < count.size(); i++)
                count[i] += count[i - 1];
                
            for (int i = n - 1; i >= 0; i--)
                sa[--count[rank[tempSA[i]]]] = tempSA[i];
            
            // Recompute ranks
            tempRank[sa[0]] = 0;
            int r = 0;
            for (int i = 1; i < n; i++) {
                if (rank[sa[i]] != rank[sa[i - 1]] ||
                    (sa[i] + gap < n ? rank[sa[i] + gap] : -1) != 
                    (sa[i - 1] + gap < n ? rank[sa[i - 1] + gap] : -1))
                    r++;
                tempRank[sa[i]] = r;
            }
            swap(rank, tempRank);
            
            if (rank[sa[n - 1]] == n - 1)
                break;
        }
    }
    
    // Build LCP array using Kasai's algorithm (O(n))
    void buildLCP(int n) {
        lcp.resize(n - 1);
        vector<int> invSA(n);
        for (int i = 0; i < n; i++)
            invSA[sa[i]] = i;
            
        int k = 0;
        for (int i = 0; i < n; i++) {
            if (invSA[i] == n - 1) {
                k = 0;
                continue;
            }
            
            int j = sa[invSA[i] + 1];
            while (i + k < n && j + k < n && s[i + k] == s[j + k])
                k++;
                
            lcp[invSA[i]] = k;
            if (k > 0)
                k--;
        }
    }
    
    const vector<int>& getSuffixArray() const {
        return sa;
    }
    
    const vector<int>& getLCP() const {
        return lcp;
    }
};

//------------------
// HEAVY-LIGHT DECOMPOSITION
//------------------

// Heavy-Light Decomposition for path queries on trees
class HeavyLightDecomposition {
private:
    vector<vector<int>> adj;
    vector<int> parent, depth, heavy, head, pos;
    vector<int> values;  // Values on vertices
    int n, timer;
    
    // Segment tree for path queries
    vector<int> tree;
    
    // DFS to compute subtree sizes and find heavy edges
    int dfs(int v) {
        int size = 1;
        int maxSubtree = 0;
        for (int u : adj[v]) {
            if (u == parent[v]) continue;
            
            parent[u] = v;
            depth[u] = depth[v] + 1;
            int subtree = dfs(u);
            
            if (subtree > maxSubtree) {
                maxSubtree = subtree;
                heavy[v] = u;
            }
            
            size += subtree;
        }
        return size;
    }
    
    // Decompose the tree into heavy paths
    void decompose(int v, int h) {
        head[v] = h;
        pos[v] = timer++;
        
        if (heavy[v] != -1)
            decompose(heavy[v], h);
            
        for (int u : adj[v]) {
            if (u != parent[v] && u != heavy[v])
                decompose(u, u);
        }
    }
    
    // Segment tree operations
    void buildTree(int node, int start, int end) {
        if (start == end) {
            tree[node] = values[start];
            return;
        }
        int mid = (start + end) / 2;
        buildTree(2 * node, start, mid);
        buildTree(2 * node + 1, mid + 1, end);
        tree[node] = max(tree[2 * node], tree[2 * node + 1]);
    }
    
    void updateTree(int node, int start, int end, int idx, int val) {
        if (start == end) {
            tree[node] = val;
            return;
        }
        int mid = (start + end) / 2;
        if (idx <= mid)
            updateTree(2 * node, start, mid, idx, val);
        else
            updateTree(2 * node + 1, mid + 1, end, idx, val);
        tree[node] = max(tree[2 * node], tree[2 * node + 1]);
    }
    
    int queryTree(int node, int start, int end, int l, int r) {
        if (r < start || end < l)
            return INT_MIN;
        if (l <= start && end <= r)
            return tree[node];
        int mid = (start + end) / 2;
        int p1 = queryTree(2 * node, start, mid, l, r);
        int p2 = queryTree(2 * node + 1, mid + 1, end, l, r);
        return max(p1, p2);
    }
    
    // Query for the maximum value on the path from u to v
    int queryPath(int u, int v) {
        int res = INT_MIN;
        while (head[u] != head[v]) {
            if (depth[head[u]] < depth[head[v]])
                swap(u, v);
            
            int pathMax = queryTree(1, 0, n - 1, pos[head[u]], pos[u]);
            res = max(res, pathMax);
            u = parent[head[u]];
        }
        
        if (depth[u] < depth[v])
            swap(u, v);
            
        int lastPath = queryTree(1, 0, n - 1, pos[v], pos[u]);
        res = max(res, lastPath);
        
        return res;
    }
    
public:
    HeavyLightDecomposition(int size, const vector<int>& vals) : n(size), values(vals) {
        adj.resize(n);
        parent.resize(n);
        depth.resize(n, 0);
        heavy.resize(n, -1);
        head.resize(n);
        pos.resize(n);
        timer = 0;
        
        tree.resize(4 * n);
    }
    
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    void build(int root = 0) {
        parent[root] = root;
        depth[root] = 0;
        dfs(root);
        decompose(root, root);
        buildTree(1, 0, n - 1);
    }
    
    void update(int vertex, int value) {
        updateTree(1, 0, n - 1, pos[vertex], value);
    }
    
    int queryMax(int u, int v) {
        return queryPath(u, v);
    }
};

//------------------
// CENTROID DECOMPOSITION
//------------------

// Centroid Decomposition of a tree
class CentroidDecomposition {
private:
    vector<vector<int>> adj;
    vector<bool> removed;
    vector<int> subtreeSize;
    vector<int> centroidParent;
    int n;
    
    // Compute subtree sizes
    void computeSubtreeSize(int v, int p) {
        subtreeSize[v] = 1;
        for (int u : adj[v]) {
            if (u != p && !removed[u]) {
                computeSubtreeSize(u, v);
                subtreeSize[v] += subtreeSize[u];
            }
        }
    }
    
    // Find centroid of the tree rooted at v with size treeSize
    int findCentroid(int v, int p, int treeSize) {
        bool isCentroid = true;
        for (int u : adj[v]) {
            if (u != p && !removed[u]) {
                if (subtreeSize[u] > treeSize / 2) {
                    isCentroid = false;
                    break;
                }
            }
        }
        
        if (isCentroid && treeSize - subtreeSize[v] <= treeSize / 2)
            return v;
            
        for (int u : adj[v]) {
            if (u != p && !removed[u]) {
                if (subtreeSize[u] > treeSize / 2)
                    return findCentroid(u, v, treeSize);
            }
        }
        
        return -1;  // Should never reach here
    }
    
    // Build centroid decomposition recursively
    void buildCentroidTree(int v, int p) {
        computeSubtreeSize(v, -1);
        int centroid = findCentroid(v, -1, subtreeSize[v]);
        
        centroidParent[centroid] = p;
        removed[centroid] = true;
        
        for (int u : adj[centroid]) {
            if (!removed[u])
                buildCentroidTree(u, centroid);
        }
        
        removed[centroid] = false;  // Restore for graph traversal
    }
    
public:
    CentroidDecomposition(int size) : n(size) {
        adj.resize(n);
        removed.resize(n, false);
        subtreeSize.resize(n);
        centroidParent.resize(n, -1);
    }
    
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    void build(int root = 0) {
        buildCentroidTree(root, -1);
    }
    
    int getLCA(int u, int v) {
        vector<bool> visited(n, false);
        while (u != -1) {
            visited[u] = true;
            u = centroidParent[u];
        }
        
        while (v != -1) {
            if (visited[v])
                return v;
            v = centroidParent[v];
        }
        
        return -1;  // Should never reach here
    }
    
    int getParent(int v) {
        return centroidParent[v];
    }
};

//------------------
// SPLAY TREE
//------------------

// Splay Tree (self-adjusting BST)
class SplayTree {
private:
    struct Node {
        int key;
        Node *left, *right, *parent;
        
        Node(int k) : key(k), left(nullptr), right(nullptr), parent(nullptr) {}
    };
    
    Node *root;
    
    // Rotate node x to the right
    void rightRotate(Node *x) {
        Node *y = x->left;
        if (y) {
            x->left = y->right;
            if (y->right)
                y->right->parent = x;
            y->parent = x->parent;
        }
        
        if (!x->parent)
            root = y;
        else if (x == x->parent->left)
            x->parent->left = y;
        else
            x->parent->right = y;
            
        if (y)
            y->right = x;
        x->parent = y;
    }
    
    // Rotate node x to the left
    void leftRotate(Node *x) {
        Node *y = x->right;
        if (y) {
            x->right = y->left;
            if (y->left)
                y->left->parent = x;
            y->parent = x->parent;
        }
        
        if (!x->parent)
            root = y;
        else if (x == x->parent->left)
            x->parent->left = y;
        else
            x->parent->right = y;
            
        if (y)
            y->left = x;
        x->parent = y;
    }
    
    // Splay operation brings a node to the root
    void splay(Node *x) {
        while (x->parent) {
            if (!x->parent->parent) {
                // Zig step
                if (x == x->parent->left)
                    rightRotate(x->parent);
                else
                    leftRotate(x->parent);
            } else if (x == x->parent->left && x->parent == x->parent->parent->left) {
                // Zig-zig step
                rightRotate(x->parent->parent);
                rightRotate(x->parent);
            } else if (x == x->parent->right && x->parent == x->parent->parent->right) {
                // Zig-zig step
                leftRotate(x->parent->parent);
                leftRotate(x->parent);
            } else if (x == x->parent->right && x->parent == x->parent->parent->left) {
                // Zig-zag step
                leftRotate(x->parent);
                rightRotate(x->parent);
            } else {
                // Zig-zag step
                rightRotate(x->parent);
                leftRotate(x->parent);
            }
        }
    }
    
    // Search for a key in the tree
    Node* search(int key) {
        Node *x = root;
        while (x && x->key != key) {
            if (key < x->key)
                x = x->left;
            else
                x = x->right;
        }
        
        if (x)
            splay(x);
        return x;
    }
    
    // Insert a key into the tree
    void insert(int key) {
        Node *z = new Node(key);
        Node *y = nullptr;
        Node *x = root;
        
        while (x) {
            y = x;
            if (z->key < x->key)
                x = x->left;
            else
                x = x->right;
        }
        
        z->parent = y;
        if (!y)
            root = z;
        else if (z->key < y->key)
            y->left = z;
        else
            y->right = z;
            
        splay(z);
    }
    
    // Find the node with the minimum key in the subtree rooted at x
    Node* minimum(Node *x) {
        while (x->left)
            x = x->left;
        return x;
    }
    
    // Replace node u with node v
    void replace(Node *u, Node *v) {
        if (!u->parent)
            root = v;
        else if (u == u->parent->left)
            u->parent->left = v;
        else
            u->parent->right = v;
            
        if (v)
            v->parent = u->parent;
    }
    
    // Delete a key from the tree
    void remove(int key) {
        Node *z = search(key);
        if (!z)
            return;
            
        if (!z->left) {
            replace(z, z->right);
        } else if (!z->right) {
            replace(z, z->left);
        } else {
            Node *y = minimum(z->right);
            if (y->parent != z) {
                replace(y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }
            replace(z, y);
            y->left = z->left;
            y->left->parent = y;
        }
        
        delete z;
    }
    
    void inOrderTraversal(Node *x, vector<int>& result) {
        if (x) {
            inOrderTraversal(x->left, result);
            result.push_back(x->key);
            inOrderTraversal(x->right, result);
        }
    }
    
public:
    SplayTree() : root(nullptr) {}
    
    void insert(int key) {
        if (!root)
            root = new Node(key);
        else
            insert(key);
    }
    
    bool find(int key) {
        return search(key) != nullptr;
    }
    
    void remove(int key) {
        remove(key);
    }
    
    vector<int> inOrder() {
        vector<int> result;
        inOrderTraversal(root, result);
        return result;
    }
};

//------------------
// LINK-CUT TREE
//------------------

// Link-Cut Tree for dynamic forest operations
class LinkCutTree {
private:
    struct Node {
        Node *left, *right, *parent;
        bool revert;
        int value;
        
        Node(int v = 0) : left(nullptr), right(nullptr), parent(nullptr),
                         revert(false), value(v) {}
    };
    
    vector<Node*> nodes;
    
    // Check if node x is a root of its splay tree
    bool isRoot(Node *x) {
        return !x->parent || (x->parent->left != x && x->parent->right != x);
    }
    
    // Push down lazy propagation
    void pushDown(Node *x) {
        if (x->revert) {
            x->revert = false;
            swap(x->left, x->right);
            if (x->left)
                x->left->revert = !x->left->revert;
            if (x->right)
                x->right->revert = !x->right->revert;
        }
    }
    
    // Rotate node x to the right
    void rightRotate(Node *x) {
        Node *y = x->left;
        x->left = y->right;
        if (y->right)
            y->right->parent = x;
        y->parent = x->parent;
        
        if (!x->parent)
            ; // x was root
        else if (x == x->parent->left)
            x->parent->left = y;
