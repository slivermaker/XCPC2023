## 数据结构

#### 主方法

```java
import java.io.*;

import java.util.*;



public class Main {


  public static void main(String[] ac) {

    ac in=new ac();

    

    

    in.flush();in.close();

  }
```



#### 字典树



```java
  //字典树插入搜查

  static class Trie{

    class Node{

      boolean end;Node next[];

      Node(){next=new Node [26];}

    }

    Node root=new Node();

    

    void insert(String s){

      Node tmp =root;

      for(char ch:s.toCharArray()){

        int idx=ch-'a';

        if(tmp.next[idx]==null)tmp.next[idx]=new Node();

        tmp=tmp.next[idx];

      }tmp.end=true;

    }

    boolean search(String w){

      Node tmp=root;

      for(char ch:w.toCharArray()){

        if(tmp.next[ch-'a']==null)return false;

        tmp=tmp.next[ch-'a'];

      }

      return tmp.end;

    }

    boolean startsWith(String w){

      Node tmp=root;

      for(char ch:w.toCharArray()){

        if(tmp.next[ch-'a']==null)return false;

        tmp=tmp.next[ch-'a'];

      }

      return true;

    }

  }
```

#### kmp

```Java
  //kmp -> indexof

  static class kmp{

    int next[];String s;String txt;

    kmp(String s,String txt){next=new int[s.length()];this.s=s;this.txt=txt;}

    void getnext(){

     for(int j=1,k=0;j<next.length;j++){

      while(k>0 && s.charAt(j)!=s.charAt(k))k=next[k-1];

      if(s.charAt(j)==s.charAt(k))k++;

      next[j]=k;

     }

    }

    int idx(){

     for(int i=0,j=0;i<txt.length();i++){

      while(j==s.length()||(j>0 && txt.charAt(i)!=s.charAt(j)))j=next[j-1];

      if(s.charAt(j)==txt.charAt(i))j++;

      if(j==s.length())return i-j+1;

     }return -1;

    }

   }

  
```

  

####   并查集

 

```java
 static class DSU{

    int id[];int rank[];

    DSU(int n){

      id=new int[n+1];rank=new int[n+1];

      for(int i=0;i<=n;i++){id[i]=i;rank[i]=1;}

    }

    int find(int x){

     if(x==id[x])return x;

      return id[x]=id[find(id[x])];

    }

    void union(int x,int y){

      int xroot=find(x);int yroot=find(y);

      if(xroot!=yroot){

        if(rank[y]<=rank[x])id[yroot]=xroot;

        else id[xroot]=yroot;

        if(rank[xroot]==rank[yroot])xroot++;

      }

    }

  }

  //反转一个数

  int rev(int x) {

    int y = 0;

    for (; x > 0; x /= 10) {

      y = y * 10 + x % 10;

    }

    return y;

  }

  //快读用 flush

  static class ac extends PrintWriter{

    StringTokenizer st;

    BufferedReader br;

    ac(){this(System.in,System.out);}

    ac(InputStream i,OutputStream o){

      super(o);br=new BufferedReader(new InputStreamReader(i));

    }

    String next(){

      try{while(st==null||!st.hasMoreTokens())st=new StringTokenizer(br.readLine());

      return st.nextToken();}catch (Exception e){}

      return null;

    }

    int nextint(){return Integer.parseInt(next());}

    long nextlong(){return Long.parseLong(next());}

  }

}
```

#### 树状数组

```java
    static class BIT{//binary indexed tree
        int nums[],tree[],n;//部分写法是填一个0号元素"0"这样add就不用k+1了
        BIT(int []ns){
            nums=ns;n=ns.length;
            tree=new int[n]; 
            for(int i=0;i<n;i++)add(i,nums[i]);
        }
        void update(int k,int x){//nums[k]=x
            add(k,x-nums[k]);
        }
        void add(int k,int x){//nums[k]+=x
            for(int i=k+1;i<=n;i+=lowbit(i)){
                tree[i-1]+=x;
            }
        }
        int sum(int l,int r){
            return presum(r)-presum(l-1);

        }
        int presum(int k){
            int ans=0;
            for(int i=k+1;i>0;i-=lowbit(i)){
                ans+=tree[i-1];
            }return ans;
        }
        

        int lowbit(int x){return (x)&(-x);}
    }

    static class BITp{//p了个区间修改
        int diff[],tree[],n;
        BITp(int []nums){
            n=nums.length;//别加上int什么的给替换了
            diff=new int[n];tree=new int[n];
            diff[0]=nums[0];
            for(int i=1;i<n;i++)diff[i]=nums[i]-nums[i-1];
            for(int i=0;i<n;i++)add(i,diff[i]);    
        }
        void update(int l,int r,int x){//令l-r上所有元素加上x
            add(l,x);
            add(r+1,-x);
        }
        void add(int k,int x){
            for(int i=k+1;i<=n;i+=lowbit(i)){
                tree[i-1]+=x;
            }
        }
        int presum(int k){
            int ans=0;
            for(int i=k+1;i>0;i-=lowbit(i))ans+=tree[i-1];
            return ans;
        }
        int lowbit(int x){return x & -x;}
    }
    static class BITpp{//pp了个区改区查
        int diff[],tree[],helpertree[];
        int n;
        BITpp(int[] nums){
            n=nums.length;diff=new int [n];tree=new int[n];helpertree=new int [n];
            diff[0]=nums[0];
            for(int i=1;i<n;i++)diff[i]=nums[i]-nums[i-1];
            for(int i=0;i<n;i++){
                add(tree,i,diff[i]);
                add(helpertree,i,i*diff[i]);
            }
        }
        void update(int l,int r,int x){
            add(tree,l,x);
            add(tree,r+1,-x);
            add(helpertree,l,l*x);
            add(helpertree,r+1,(r+1)*(-x));
        }
        void add(int tmptree[],int k,int x){
            for(int i=k+1;i<=n;i+=lowbit(i)){
                tmptree[i-1]+=x;
            }
        }
        int sum(int l,int r){
            int sum1=l*presum(tree, l-1)-presum(helpertree, l-1);
            int sum2=(r+1)*presum(tree, r)-presum(helpertree, r);
            return sum2-sum1;
        }
        int presum(int tmptree[],int k){//差分数组的前缀和
            int ans=0;
            for(int i=k+1;i>0;i-=lowbit(i)){
                ans+=tmptree[i-1];
            }return ans;
        }
        int lowbit(int x){
            return x&-x;
        }
    }
    static class RMBIT{//区间最值
        int tree[];
        int n,nums[];
        RMBIT(int ns[]){
            int n=ns.length;
            tree=new int[n+1];
            nums=new int[n+1];
            for(int i=1;i<=n;i++){
                nums[i]=ns[i-1];
                update(i,nums[i]);
            }
        }
        void update(int k,int x){
            while(k<=n){
                tree[k]=x;
                for(int i=1;i<lowbit(k);i<<=1){
                    tree[k]=Math.max(tree[k],tree[k-1]);
                }x+=lowbit(x);
            }
        }
        int rmax(int l,int r){
            int ans=0;
            while(l<=r){
                ans=Math.max(ans,nums[r]);
                r--;
                while(r-l>lowbit(r)){
                    ans=Math.max(ans,tree[r]);
                    r-=lowbit(r);
                }
            }
            return ans;
        }
        int lowbit(int x){return x&-x;}
    }

```

#### 线段树

```java
 static class SegmentTreeBasic {
	int[] nums, treeSum, treeMin, treeMax;// 所有操作都将在log级别完成
	int n;

	public SegmentTreeBasic(int[] ns) {
		this.nums = ns;
		this.n = ns.length;
		this.treeSum = new int[4 * n];
		this.treeMin = new int[4 * n];
		this.treeMax = new int[4 * n];
		build(0, n - 1, 1);
	}

	public void add(int i, int x) { // 单点修改(驱动): nums[i] += x
		add(i, x, 0, n - 1, 1);
	}

	public void update(int i, int x) {// 单点修改(驱动): nums[i] = x
		update(i, x, 0, n - 1, 1);
	}

	public int query(int i) { // 单点查询 (驱动): 查询 nums[i]
		return query(i, 0, n - 1, 1);
	}

	public int sum(int l, int r) { // 区间查询(驱动): nums[l]~nums[r]之和
		return sum(l, r, 0, n - 1, 1);
	}

	public int min(int l, int r) { // 区间查询 (驱动): 查询[l,r]中的最小值
		return min(l, r, 0, n - 1, 1);
	}

	public int max(int l, int r) { // 区间查询 (驱动): 查询[l,r]中的最大值
		return max(l, r, 0, n - 1, 1);
	}

	// 单点查询 (具体): 查询 nums[i]，尾递归
	private int query(int idx, int s, int t, int i) {
		if (s == t)
			return treeSum[i];
		int c = s + (t - s) / 2;
		if (idx <= c)
			return query(idx, s, c, i * 2);
		else
			return query(idx, c + 1, t, i * 2 + 1);
	}

	// 单点修改: nums[idx] += x
	private void add(int idx, int x, int s, int t, int i) {
		if (s == t) {
			treeSum[i] += x; // 增量更新
			treeMin[i] += x; // 增量更新
			treeMax[i] += x; // 增量更新
			return;
		}
		int c = s + (t - s) / 2;
		if (idx <= c)
			add(idx, x, s, c, i * 2);
		else
			add(idx, x, c + 1, t, i * 2 + 1);
		pushUpSum(i);
		pushUpMin(i);
		pushUpMax(i);
	}

	// 单点修改: nums[idx] = x
	private void update(int idx, int x, int s, int t, int i) {
		if (s == t) {
			treeSum[i] = x; // 覆盖更新
			treeMin[i] = x; // 覆盖更新
			treeMax[i] = x; // 覆盖更新
			return;
		}
		int c = s + (t - s) / 2;
		if (idx <= c)
			update(idx, x, s, c, i * 2);
		else
			update(idx, x, c + 1, t, i * 2 + 1);
		pushUpSum(i);
		pushUpMin(i);
		pushUpMax(i);
	}

	// 区间查询: nums[l]~nums[r]之和
	private int sum(int l, int r, int s, int t, int i) {
		if (l <= s && t <= r)
			return treeSum[i];
		int c = s + (t - s) / 2, sum = 0;
		if (l <= c)
			sum += sum(l, r, s, c, i * 2); // 递归累加目标区间落在c左侧(含c)的区间和
		if (r > c)
			sum += sum(l, r, c + 1, t, i * 2 + 1); // 递归累加目标区间落在c右侧的区间和
		return sum;
	}

	// 区间查询: 查询[l,r]中的最小值
	private int min(int l, int r, int s, int t, int i) {
		if (l <= s && t <= r)
			return treeMin[i];
		int c = s + (t - s) / 2, lmin = Integer.MAX_VALUE, rmin = Integer.MAX_VALUE;
		if (l <= c)
			lmin = min(l, r, s, c, i * 2);
		if (r > c)
			rmin = min(l, r, c + 1, t, i * 2 + 1);
		return Math.min(lmin, rmin);
	}

	// 区间查询: 查询[l,r]中的最大值
	private int max(int l, int r, int s, int t, int i) {
		if (l <= s && t <= r)
			return treeMax[i];
		int c = s + (t - s) / 2, lmax = Integer.MIN_VALUE, rmax = Integer.MIN_VALUE;
		if (l <= c)
			lmax = max(l, r, s, c, i * 2);
		if (r > c)
			rmax = max(l, r, c + 1, t, i * 2 + 1);
		return Math.max(lmax, rmax);
	}

	// 构建线段树(tree数组)
	private void build(int s, int t, int i) {
		if (s == t) { // s: start,nums当前区间起点下标，t: terminal,nums当前结点区间末尾下标
			treeSum[i] = nums[s];
			treeMin[i] = nums[s];
			treeMax[i] = nums[s];
			return;
		}
		int c = s + (t - s) / 2;
		build(s, c, i * 2);
		build(c + 1, t, i * 2 + 1);
		pushUpSum(i);
		pushUpMin(i);
		pushUpMax(i);
	}

	// pushUpSum: 更新 treeSum[i]
	private void pushUpSum(int i) {
		treeSum[i] = treeSum[i * 2] + treeSum[i * 2 + 1];
	}

	// pushUpMin: 更新 treeMin[i]
	private void pushUpMin(int i) {
		treeMin[i] = Math.min(treeMin[i * 2], treeMin[i * 2 + 1]);
	}

	// pushUpMax: 更新 treeMax[i]
	private void pushUpMax(int i) {
		treeMax[i] = Math.max(treeMax[i * 2], treeMax[i * 2 + 1]);
	}
}

  



static class SegmentTree {// 单点修改，区间查询和 求最值时为O（n）
        int nums[], tree[], n;
    
        SegmentTree(int[] ns) {
            nums = ns;
            n = ns.length;
            tree = new int[4 * n];
    //		build(0, n - 1, 1);
        }
    
        void build(int s, int t, int i) {
            if (s == t) {
                tree[i] = nums[s];
                return;
            }
            int c = s + (t - s) / 2;
            build(s, c, i * 2);
            build(c + 1, t, i * 2 + 1);
            tree[i] = tree[i * 2] + tree[i * 2 + 1];
        }
    
        void add(int i, int x) {// nums[i]+=x驱动
            add(i, x, 0, n - 1, 1);
        }
    
        void update(int i, int x) {
            add(i, x - nums[i], 0, n - 1, 1);
            nums[i] = x;
        }
    
        int query(int i) {
            return nums[i];
        }
    
        int sum(int l, int r) {
            return sum(l, r, 0, n - 1, 1);
        }
    
        int max(int l, int r) {
            return max(l, r, 0, n - 1, 1);
        }
    
        void add(int id, int x, int s, int t, int i) {
            if (s == t) {
                tree[i] += x;
                return;
            }
            int c = s + (t - s) / 2;
            if (id <= c)
                add(id, x, s, c, i * 2);
            else
                add(id, x, c + 1, t, i * 2 + 1);
            tree[i] =Math.max(tree[i*2],tree[i*2+1]);
        }
    
        int sum(int l, int r, int s, int t, int i) {
            if (l <= s && t <= r)
                return tree[i];
            int c = s + (t - s) / 2, sum = 0;
            if (l <= c)
                sum += sum(l, r, s, c, i * 2); // 递归累加目标区间落在c左侧(含c)的区间和
            if (r > c)
                sum += sum(l, r, c + 1, t, i * 2 + 1); // 递归累加目标区间落在c右侧的区间和
            return sum;
    
        }
    
        int max(int l, int r, int s, int t, int i) {
            if (l <= s && t <= r)
                return tree[i];
            int c = s + (t - s) / 2;
            int lmax = Integer.MIN_VALUE, rmax = Integer.MIN_VALUE;
            if (l <= c)
                lmax = max(l, r, s, c, i * 2);
            if (r > c)
                rmax = max(l, r, c + 1, t, i * 2 + 1);
            return Math.max(lmax, rmax);
        }
    
        int min(int l, int r, int s, int t, int i) {
            if (s == t)
                return tree[s];
            int c = s + (t - s) / 2;
            int lmin = Integer.MAX_VALUE, rmin = Integer.MAX_VALUE;
            if (l <= c)
                lmin = min(l, r, s, c, i * 2);
            if (r > c)
                lmin = min(l, r, s, c + 1, i * 2 + 1);
            return Math.min(lmin, rmin);
        }
```





## 算法

#### 构造篇

[分数小于1900的构造题](https://codeforces.com/problemset?order=BY_SOLVED_DESC&tags=constructive+algorithms%2C1000-1900)

以下是题目简述及思路：



#### 动态规划篇

##### [区间 DP](https://github.com/SharingSource/LogicStack-LeetCode/wiki/区间-DP)

大多数情况dp[i] [j]先填充



###### 状态转移方程

思路一：

以下三种是从空白串刷到目标串所用次数最小的情况，状态转移方程方程不唯一

```java
for (int len = 2; len <= n; ++len) {//固定遍历长度，1的情况已经填充完毕，最长的情况为n
	for (int i = 1; i + len - 1  <= n; ++i) {//考虑下标，从一开始，加到i+len-1之后他们之间才覆盖len个元素
		int j = i + len - 1;
		for (int k = i; k < j; ++k) {//k是分割点
			dp[i,j] = max/min(dp[i,j], dp[i,k] + dp[k+1, j] + cost);
		}
	}
}
//最小打印次数题解，当len=1开始时包括了dp[i][i]==1的情况，而且这道题适合这样开
class Solution {
    public int strangePrinter(String s) {
        int n = s.length();
        int[][] f = new int[n + 1][n + 1];
        for (int len = 1; len <= n; len++) {
            for (int l = 0; l + len - 1 < n; l++) {
                int r = l + len - 1;
                f[l][r] = f[l + 1][r] + 1;
                for (int k = l + 1; k <= r; k++) {
                    if (s.charAt(l) == s.charAt(k)) {
                        f[l][r] = Math.min(f[l][r], f[l][k - 1] + f[k + 1][r]);
                    }
                }
            }
        }
        return f[0][n - 1];
    }
}
class Solution {
    public int strangePrinter(String s) {
        int n = s.length();
        char[] ch=(s).toCharArray();
        int[][] f = new int[n + 2][n + 2];
        for(int i=0;i<=n;i++)f[i][i]=1;
        for (int len = 2; len <= n; ++len) {//固定遍历长度，1的情况已经填充完毕，最长的情况为n
            for (int i = 0; i + len - 1  <n; ++i) {//考虑下标，从一开始，加到i+len-1之后他们之间才覆盖len个元素
                int j = i + len - 1;
                f[i][j]=f[i+1][j]+1;
                for (int k = i+1; k <= j; ++k) {//k是分割点这道题里k=i+1,否则下标超出
                if(ch[i]==ch[k])
                    f[i][j] = Math.min(f[i][j], f[i][k-1] + f[k+1][j]);
                }
            }
        }
        return f[0][n-1];
    }
}
class Solution {
    int[][] f=new int [105][105];
    public int strangePrinter(String s) {
        int n = s.length();
        char[] ch=(" "+s+"  ").toCharArray();
        
        for(int i=1;i<=n;i++)f[i][i]=1;
        for (int len = 2; len <= n; ++len) {
            for (int i = 1; i + len - 1  <=n; ++i) {
                int j = i + len - 1;
                f[i][j]=Integer.MAX_VALUE;
                if(ch[i]==ch[j])f[i][j]=f[i+1][j];//如果两端相等则一次就可以
                else for(int k=i;k<j;++k) f[i][j]=Math.min(f[i][j],f[i][k]+f[k+1][j]);//如果不相等就分开刷
                }
            } return f[1][n];
        }       

    }

```

###### 例题[NOIP2006 提高组] 能量项链

在 Mars 星球上，每个 Mars 人都随身佩带着一串能量项链。在项链上有 $N$ 颗能量珠。能量珠是一颗有头标记与尾标记的珠子，这些标记对应着某个正整数。并且，对于相邻的两颗珠子，前一颗珠子的尾标记一定等于后一颗珠子的头标记。因为只有这样，通过吸盘（吸盘是 Mars 人吸收能量的一种器官）的作用，这两颗珠子才能聚合成一颗珠子，同时释放出可以被吸盘吸收的能量。如果前一颗能量珠的头标记为 $m$，尾标记为 $r$，后一颗能量珠的头标记为 $r$，尾标记为 $n$，则聚合后释放的能量为 $m \times r \times n$（Mars 单位），新产生的珠子的头标记为 $m$，尾标记为 $n$。

需要时，Mars 人就用吸盘夹住相邻的两颗珠子，通过聚合得到能量，直到项链上只剩下一颗珠子为止。显然，不同的聚合顺序得到的总能量是不同的，请你设计一个聚合顺序，使一串项链释放出的总能量最大。

例如：设 $N=4$，$4$ 颗珠子的头标记与尾标记依次为 $(2,3)(3,5)(5,10)(10,2)$。我们用记号 $\oplus$ 表示两颗珠子的聚合操作，$(j \oplus k)$ 表示第 $j,k$ 两颗珠子聚合后所释放的能量。则第 $4$ 、 $1$ 两颗珠子聚合后释放的能量为：

$(4 \oplus 1)=10 \times 2 \times 3=60$。

这一串项链可以得到最优值的一个聚合顺序所释放的总能量为：

$((4 \oplus 1) \oplus 2) \oplus 3)=10 \times 2 \times 3+10 \times 3 \times 5+10 \times 5 \times 10=710$。

输入格式

第一行是一个正整数 $N$（$4 \le N \le 100$），表示项链上珠子的个数。第二行是 $N$ 个用空格隔开的正整数，所有的数均不超过 $1000$。第 $i$ 个数为第 $i$ 颗珠子的头标记（$1 \le i \le N$），当 $i<N$ 时，第 $i$ 颗珠子的尾标记应该等于第 $i+1$ 颗珠子的头标记。第 $N$ 颗珠子的尾标记应该等于第 $1$ 颗珠子的头标记。

至于珠子的顺序，你可以这样确定：将项链放到桌面上，不要出现交叉，随意指定第一颗珠子，然后按顺时针方向确定其他珠子的顺序。

输出格式

一个正整数 $E$（$E\le 2.1 \times 10^9$），为一个最优聚合顺序所释放的总能量。

样例 #1

样例输入 #1

```
4
2 3 5 10
```

样例输出 #1

```
710
```

提示

NOIP 2006 提高组 第一题

```cpp
#include <iostream> 
#define MAXN 202
#define MAX(a,b) ((a)>(b)?(a):(b))
using namespace std;
int n,a[MAXN],dp[MAXN][MAXN],ans;
int main(){
    cin>>n;
    for(int i=1;i<=n;i++)
        cin>>a[i],a[i+n]=a[i];
    for(int len=2;len<=n;len++)//枚举区间长度
        for(int i=1;i+len-1<n*2;i++){//枚举区间起点
            int j=i+len-1;//区间终点
            for(int k=i;k<j;k++)//枚举决策
                dp[i][j]=MAX(dp[i][j], dp[i][k]+dp[k+1][j]+a[i]*a[k+1]*a[j+1]);
        }
    for(int i=1;i<=n;i++)//枚举可能的答案
        ans=MAX(dp[i][i+n-1],ans);
    cout<<ans;
    return 0;
}
```









##### 树型dp

###### 储存树的方法

```java
ArrayList<ArrayList<Integer>> degrees = new ArrayList<>();
for (int t = 0;t < n;t++) {
    degrees.add(new ArrayList<Integer>());
}
for(int[] edge: edges) {
    degrees.get(edge[0]).add(edge[1]);
    degrees.get(edge[1]).add(edge[0]);
}
```

```java
Map<Integer,List<Integer>>graph = new HashMap<>();//理论上更节省空间，但是顺序拿取情况下耗时巨长，特定会快一些
int[] inDegree = new int[n];
for(int[] edge:edges){
    int a = edge[0];
    int b = edge[1];
    graph.putIfAbsent(a,new ArrayList<>());
    graph.putIfAbsent(b,new ArrayList<>());
    graph.get(a).add(b);
    graph.get(b).add(a);
    ++inDegree[a];
    ++inDegree[b];
}

```

###### 链式向前星储存

```java
int N = 10; // 最大节点数
int M = 100; // 最大边数

// 链式前向星存储结构，其中edge和to都是可选的，按实际情况选择
// 也可以将edge和to合入一个类中
int[] head = new int[N]; // head[n] 表示编号n的节点最新一条边的索引
int[] edge = new int[M]; // edge[i] 表示索引为i的边的权重(长度)
int[] to = new int[M];   // to[i] 表示索引为i的边的终点索引
int[] next = new int[M]; // next[i] 表示索引为i的边 下一条同起点的边的索引，用于找到同起点的下一条边
int i = 0; // 边索引

Arrays.fill(head, -1);   // 先设置所有编号节点都没有边(-1表示该节点没有边)

// 添加一条新的边，索引为i，起点为a，终点为b
void add(int a, int b) {
    to[i] = b;          // 索引为i的边终点为b
    next[i] = head[a];  // 该边下一条同起点边的索引为head[a]
    head[a] = i++;      // 更新起点a最新的一条边索引为i
}

// 遍历所有从a发出的边
// 从head[a]取出a节点最新一条边的索引idx，开始遍历。
// 每次通过next[idx]获取以该节点为起点的下一条边的索引
// 直到下一条边的索引为-1，即没有下一条边
for (int idx = head[a]; idx != -1; idx = next[idx]) {
		// 取出该索引对应的边的终点
    int end = to[idx];
    // 取出该索引对应的边的权重
    int val = edge[idx];
}

```

类执行

```java
//类执行
private static Edge[] edges ;//边集数组，存放所有的边
private static int cnt = 0; //记录边的下标，比如，如果边的编号从零开始，那么第1条边的下标就是0
private static int[] head;//头结点数组，

//    建立边集
public static class Edge{
    int to;
    int w;
    int next;
}

public static void add(int u,int v,int w){
    edges[cnt] = new Edge();//创建一条边将其放入边集数组
    edges[cnt].to = v;
    edges[cnt].w = w;//权重
    edges[cnt].next = head[u];
    head[u] = cnt++;
}
public static void bfs(int node_num){
//  标记数组，用于标记没有访问过的结点    
    boolean[] visited = new boolean[node_num+1];
    Queue<Integer> queue = new LinkedList<>();
    queue.offer(1);
    while (!queue.isEmpty()){
        int u = queue.poll();
        System.out.print(u+" ");
//        从某个点开始，访问这个点出发的所有边
        for (int i = head[u]; i != -1; i = edges[i].next) {
            int f = edges[i].to;
            if (!visited[f]){
                queue.offer(f);
                visited[f] = true;
            }
        }
    }
}

```

## 题目

### 牛客多校

###### 例题1贪吃的派蒙（贪心模拟

时间限制：C/C++ 1秒，其他语言2秒  
空间限制：C/C++ 262144K，其他语言524288K  
64bit IO Format: %lld  

题目描述

在遥远的提瓦特大陆上，正在筹备一年一度的羽球节，猎鹿人餐厅为犒劳认真筹备的众人，准备了K份甜甜花酿鸡供大家排队领取。

在每一次的排队中，编号为i的角色领取上限为Ai，这意味着他可以领取的甜甜花酿鸡在\[1-Ai\]范围内。当一个角色领完本次的甜甜花酿鸡，他/她就会回到队列的末尾，直到所有甜甜花酿鸡都被吃完为止。当轮到一个角色领取时，如果所有的甜甜花酿鸡都被领完，那么他/她就要帮大家刷盘子。

贪吃的派蒙每次都吃固定的Ax个甜甜花酿鸡(如果剩下的甜甜花酿鸡的数量比Ax小，那么他就把剩下的都吃完)。我们很容易找到派蒙的编号，Ax比其他所有的Ai都要大。大家都想让派蒙最后留下来刷盘子，请你写一个程序来判断这是否可能。

输入描述:

```
第一行包含一个整数T(1≤T≤100)，表示有T组测试数据。接下来每组测试数据包含两行。第一行包含两个整数N和K(2≤N≤10^5,0≤K≤10^8)，分别表示人数和甜甜花酿鸡的数量。第二行包含一个整数Ai(1≤Ai≤10^9)，表示队列中编号为i的角色可以领取甜甜花酿鸡的最大数量。始终只有一个最大的Ax。
```

输出描述:

```
如果大家能找到一种方案让派蒙刷盘子，那么输出“YES”。否则输出“NO”。
```

```cpp
#include<iostream>
using namespace std;
#include<vector>
#include<algorithm>
#include<math.h>
#include<set>
#include <random>
#include<numeric>
#include<string>
#include<string.h>
#include<iterator>

#include<map>
#include<unordered_map>
#include<stack>
#include<list>
#include<queue>
#include<iomanip>
#include<bitset>

//#pragma GCC optimize(2)
//#pragma GCC optimize(3)

#define endl '\n'
#define int ll
#define PI acos(-1)
#define INF 0x3f3f3f3f
typedef long long ll;
typedef unsigned long long ull;
typedef pair<ll, ll>PII;
const int N = 1e5+50, MOD = 1000639;

int a[N];
void solve() 
{
    int n, k, sum = 0, mx = 0, idx = 0;
    cin >> n >> k;
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
        //找派蒙
        if (mx < a[i])
        {
            
            mx = a[i];
            idx = i;
        }
        //计算所有人的食量
        sum += a[i];
    }
    int ans = 0;
    //求派蒙前面所有人的食量
    for (int i = 1; i < idx; i++)ans += a[i];
    //只要前面的人拿不完（最少拿idx-1个）
    //然后最多情况下到派蒙正好吃完:ans+mx
    //鸡的数量在这之间就是派蒙洗碗
    if (k > idx - 1 && k <= ans + mx)
    {
        cout << "YES" << endl;
        return;
    }
    //减去前面的数量
    k -= ans + mx;
    //取模，快进吃鸡的过程
    k %= sum;
    //同上情况，不过此时第一个吃的是派蒙后面的第一个人，当n-1个人都拿了后才到派蒙
    if (k > n - 1 && k < sum)cout << "YES" << endl;
    else cout << "NO" << endl;
}
signed main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int t = 1;
    cin >> t;
    while (t--)
    {
        solve();
    }
    return 0;
}
```



