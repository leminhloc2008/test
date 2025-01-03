#include <bits/stdc++.h>
using namespace std;

const int MAXN = 1e6 + 5;
const int MOD = 1e9 + 7;

long long N, Hash[MAXN], Pow[MAXN];

long long getHash(long long l, long long r) {
    return (Hash[r] - Pow[r - l + 1] * Hash[l - 1] % MOD + MOD) % MOD;
}

int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    string s;
    cin >> s;
    N = s.size();
    s = ' '  + s;
    Pow[0] = 1;
    for (int i = 1; i <= N; i++) {
        Hash[i] = (31 * Hash[i - 1] + s[i] - 'a' + 1) % MOD;
        Pow[i] = 31 * Pow[i - 1] % MOD;
    }
    cout << getHash(1, 3);
    return 0;
}
