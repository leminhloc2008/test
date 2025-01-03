
#include <bits/stdc++.h>
using namespace std;

// n2loclm
#define ll long long
#define st string
#define de double

#define vel vector<ll>
#define ves vector<st>
#define veb vector<bool>
#define pb push_back
#define pob pop_back

#define mall map<ll, ll>
#define masl map<st, ll>
#define malb map<ll, bool>
#define masb map<st, bool>
#define sll set<ll>
#define fi first
#define se second

#define fo(i, a, b) for (ll i = a; i < b; i++)
#define fob(i, a, b) for (ll i = a - 1; i >= b; i--)
#define foe(i, a, b) for (ll i = a; i <= b; i++)
#define foeb(i, a, b) for (ll i = a; i >= b; i--)

#define fod(i, a, b) for (de i = a; i < b; i++)

#define sorta(a, b) sort(a + b, a + b + 1);
#define sortv(v) sort(v.begin(), v.end());

const ll mod = 1e9 + 7;

// start template
// author: n2loclm
long long a[10000005];
long long b[10000005];
// kiem tra so nguyen to
int ktNguyenTo(long long n)
{
    if (n <= 1)
        return 0;
    if (n <= 3)
        return 1;
    if (n % 2 == 0 || n % 3 == 0)
        return 0;
    for (long long i = 5; i * i <= n; i += 6)
    {
        if (n % i == 0 || n % (i + 2) == 0)
            return 0;
    }
    return 1;
}
// to hop
long long tohop(long long n, long long m)
{
    long long t = 1;
    for (int i = 1; i <= m; i++)
    {
        t = t * (n - m + i) / i;
    }
    return t;
}
// sang nguyen to
long long nguyento[2000005];
long long m = int(1e9 + 7);
void sang()
{
    nguyento[0] = nguyento[1] = 1;
    for (int i = 2; i * i <= m; i++)
    {
        if (nguyento[i] == 0)
        {
            for (int j = i * i; j <= m; j += i)
            {
                nguyento[j] = 1;
            }
        }
    }
}

vector<long long> prime;
// sang vector
vector<long long> sangVector()
{
    vector<bool> isPrime(m + 1, true);
    isPrime[0] = isPrime[1] = false;
    for (int i = 2; i * i <= m; i++)
    {
        if (isPrime[i])
        {
            for (int j = i * i; j <= m; j += i)
            {
                isPrime[j] = false;
            }
        }
    }
    prime.push_back(2);
    for (int i = 3; i <= m; i += 2)
    {
        if (isPrime[i])
        {
            prime.push_back(i);
        }
    }
    return prime;
}
// doi string sang so
long long stringToInt(string s)
{
    long long t = 0;
    for (int i = 0; i <= s.size() - 1; i++)
        t = t * 10 + (s[i] - 48);
    return t;
}
long long tinhTich(long long a, long long n, long long m)
{
    if (n == 0)
        return 0;
    if (n == 1)
        return a;
    long long t = tinhTich(a, n / 2, m) % m;
    if (n % 2 == 0)
        return t * 2 % m;
    if (n % 2 == 1)
        return (t * 2 % m + a) % m;
}

long long tinhMu(long long arr, long long n, long long m)
{
    if (n == 0)
        return 1;
    if (n == 1)
        return arr;
    long long t = tinhMu(arr, n / 2, m) % m;
    long long tich = tinhTich(t, t, m);
    if (n % 2 == 0)
        return tich % m;
    if (n % 2 == 1)
        return tinhTich(tich, arr, m);
}
// tong chu so cua n
int tongcs(long long n)
{
    long long s = 0;
    while (n > 0)
        (s = s + n % 10, n = n / 10);
    return s;
}
// kiem tra nguyen to fermat
long long kiemTraFermat(long long n, long long k = 50)
{
    if (n < 4)
    {
        return n == 2 || n == 3;
    }
    if (n != 2 && n % 2 == 0)
    {
        return false;
    }
    for (int i = 1; i <= k; i++)
    {
        long long a = rand() % (n - 3) + 2;
        if (tinhMu(a, n - 1, n) != 1)
        {
            return false;
        }
    }
    return true;
}
// kiem tra nguyen to milner rabin
long long kiemTraMilnerRabin(long long n, long long k = 10)
{
    if (n < 2)
    {
        return false;
    }
    if (n != 2 && n % 2 == 0)
    {
        return false;
    }
    long long d = n - 1;
    while (d % 2 == 0)
    {
        d /= 2;
    }
    for (int i = 1; i <= k; i++)
    {
        long long a = rand() % (n - 1) + 1, t = d, modVal = tinhMu(a, t, n);
        while (t != n - 1 && modVal != 1 && modVal != n - 1)
        {
            modVal = tinhTich(modVal, modVal, n);
            t *= 2;
        }
        if (modVal != n - 1 && t % 2 == 0)
        {
            return false;
        }
    }
    return true;
}
// dem uoc
bool ktChinhPhuong(long long n)
{
    long long can = sqrt(n);
    if (can * can == n)
    {
        return true;
    }
    else
    {
        return false;
    }
}

long long demUoc(long long n)
{
    vector<long long> prime = sangVector();
    long long kq = 1;
    for (int i : prime)
    {
        if (i * i * i > n)
        {
            break;
        }
        long long dem = 0;
        while (n % i == 0)
        {
            n = n / i;
            dem++;
        }
        kq = kq * (dem + 1);
    }
    if (kiemTraMilnerRabin(n))
    {
        kq *= 2;
    }
    else if (ktChinhPhuong(n) && kiemTraMilnerRabin(sqrt(n)))
    {
        kq *= 3;
    }
    else if (n != 1)
    {
        kq *= 4;
    }
    return kq;
}
// phan tich thua so nguyen to
long long somu[2000005];

void pttsnt(long long n)
{
    sangVector();
    long long x = 0;
    while (prime[x] * prime[x] <= n && n > 1)
    {
        if (n % prime[x] == 0)
        {
            somu[prime[x]]++;
            n = n / prime[x];
        }
        else
        {
            x++;
        }
    }
    if (n != 1)
    {
        somu[n]++;
    }
}
// dem chu so cua n
int demcs(long long n)
{
    long long dem = 0;
    while (n > 0)
        (n = n / 10, dem++);
    return dem;
}
// dem uoc so
int demuocso(long long n)
{
    long long count = 0;
    for (int i = 1; i <= sqrt(n); i++)
    {
        if (n % i == 0)
        {
            count++;
            long long k = n / i;
            if (k != i)
                count++;
        }
    }
    return count;
}
// dao nguoc so
int daonguocso(long long n)
{
    long long k = 0, t = 0;
    while (n != 0)
    {
        k = n % 10;
        t = t * 10 + k;
        n /= 10;
    }
    return t;
}
// dem so chia het cho m trong doan a den b
int demChiaHet(long long a, long long b, long long m)
{
    if (a % m == 0)
    {
        return (b / m) - (a / m) + 1;
    }
    return (b / m) - (a / m);
}

// sang uoc, d=so luong uoc so, t=tong uoc so
long long d[10000005], t[10000005];

void sang()
{
    long long n = 1e6;
    for (int i = 1; i <= n; i++)
    {
        d[i] = 1;
        t[i] = 1;
    }
    for (int i = 2; i <= n; i++)
    {
        for (int j = i; j <= n; j += i)
        {
            d[j] += 1;
            t[j] += i;
        }
    }
}

// tổng đoạn con lớn nhất
int maxSubArray(long long n, long long a[])
{
    long long temp = 0, tong = -1e9;
    for (int i = 1; i <= n; i++)
    {
        temp = temp + a[i];
        if (tong < temp)
        {
            tong = temp;
        }
        if (temp < 0)
        {
            temp = 0;
        }
    }
    return tong;
}
// phan tich thua so nguyen to
void pttsnt(long long n)
{
    long long c = 2;
    while (c * c <= n && n > 1)
    {
        if (n % c == 0)
        {
            cout << n << " ";
            n /= c;
        }
        else
        {
            c++;
        }
    }
    if (n > 1)
        cout << n << " ";
}

vector <long long> graph[100005], canh[100005];
long long dp[1000005], cha[1000005], kt[1000005];

struct dinh{
    long long ver, dd;
};

struct cmp{
    bool operator() (dinh a, dinh b) {
        return a.dd > b.dd;
    }
};

priority_queue<dinh, vector<dinh>, cmp> heap_min;

void push(long long x)
{
    dinh y;
    y.ver = x;
    y.dd = dp[x];
    heap_min.push(y);
}

void dij(long long n, long long m, long long s)
{
    for (int i = 1; i <= n; i++)
    {
        dp[i] = 1e18;
        cha[i] = kt[i] = 0;
    }
    dp[s] = 0;
    cha[s] = s;
    push(s);
    while(!heap_min.empty())
    {
        dinh x = heap_min.top();
        heap_min.pop();
        long long u = x.ver;
        kt[u] = 1;
        for (int i = 0; i < graph[u].size(); i++)
        {
            long long v = graph[u][i];
            if (kt[v]) continue;
            if (dp[u] + canh[u][i] < dp[v])
            {
                dp[v] = dp[u] + canh[u][i];
                cha[v] = u;
                push(v);
            }
        }
    }
}

long long findset(long long i)
{
    while (p[i] > 0) i = p[i];
    return i;
}

long long join(long long i, long long j)
{
    long long u = findset(i), v = findset(j);
    if (u == v) return 0;
    if (u < v)
    {
        p[u] = p[u] + p[v];
        p[v] = u;
    }
    else
    {
        p[v] = p[u] + p[v];
        p[u] = v;
    }
}

long long disc[1000005], low[1000005], xoa[1000005], n, id = 0, cau[1000005], khop[10000005], con[10000005];
deque <long long> dq;

void dfs(long long u)
{
    id++;
    disc[u] = id;
    low[u] = disc[u];
    id++;
    disc[u] = id;
    low[u] = disc[u];
    for (int i = 0; i < graph[u].size(); i++)
    {
        long long v = graph[u][i];
        if (disc[v] > 0)
        {
            if (cha[u] != v) low[u] = min(low[u], disc[v]);
        }
        else
        {
            cha[u] = v;
            con[u]++;
            dfs(v);
            low[u] = min(low[u], low[v]);
        }
    }

    for 
    v
    if(disc[v] > 0)
    {
        if (cha[u] != v) low[u] = min(low[u], disc[v])''
    }
    {
        cha[u] = v;
        con[u]++;
        dfs(v_);
        low[u] = min(low[u], low[v]);
    }
    // tim tplt manh
    if (low[u] == disc[u])
    {
        while (dq.back() != u)
        {
            xoa[dq.back()] = 1;
            cout << dq.back() << " ";
            dq.pop_back();
        }
        cout << u << "\n";
        dq.pop_back();
        xoa[u] = 1;
    }
}

void caukhop()
{
    for (int i = 1; i <= n; i++)
    {
        if (cha[i] == 0)
        {
            id = 0;
            cha[i] = i;
            dfs(i);
        }
    }
    long long dcau = 0, dkhop = 0;
    unordered_map <long long, long long> cnt;
    cha[v] != v;
    u = cha[v];
    low[v] >= disc[u] && !(disc[u] == ! && con[u] < 2)
    low[v] >= disc[v] 
    for (int v = 1; v <= n; v++)
    {
        if (cha[v] != v)
        {
            long long u = cha[v];
            if (low[v] >= disc[u])
            {
                if (disc[u] == 1 && con[u] < 2) khop[u] = 0;
                else
                {
                    khop[u] = 1;
                    if (!cnt[u]) dkhop++, cnt[u]++;
                }
            }
        }
    }
    for (int v = 1; v <= n; v++)
    {
        if (cha[v] != v && low[v] >= disc[v])
        {
            cau[v] = 1;
            dcau++;
        }
    }
}
// end template

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    long long t = 1;
    while (t--)
    {
        long long n;
        cin >> n;
    }
    return 0;
}
