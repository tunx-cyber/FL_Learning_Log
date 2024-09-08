https://oi-wiki.org/contest/roadmap/

排序的基本idea：

- 选择排序，每次都选择最大或者最小的数，与边界交换。

- 冒泡排序，每次都交换出最大或者最小的数到边界。

- 插入排序，从第二个元素开始，看插入哪里，然后移动后面的数组，让自己插进去。

- 计数排序，用一个哈希表记录每一个出现次数，然后利用前缀和计算出每个数的排名，然后利用额外的数组记录原数组每个元素的应该在哪个位置，倒序操作

- 基数排序，基数排序将待排序的元素拆分为 ![k](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7) 个关键字，逐一对各个关键字排序后完成对所有元素的排序。

- 快速排序，第一步要是要把数列分成两个部分，然后保证前一个子数列中的数都小于后一个子数列中的数。可以随机找一个作为参考，比如最左或者最右，维护两个指针，保证先对于pivot的大小关系。

- 归并排序，使用额外数组，根据大小关系合并两个数组。重复上述过程直到 `a[i]` 和 `b[j]` 有一个为空时，将另一个数组剩下的元素放入 `c[k]`。

- 堆排序，首先建立大顶堆，然后将堆顶的元素取出，作为最大值，与数组尾部的元素交换，并维持残余堆的性质；之后将堆顶的元素取出，作为次大值，与数组倒数第二位元素交换，并维持残余堆的性质；

- 桶排序，额外一个和原数组长的数组，作为桶，对于重复元素，每个桶也是个数组，对这个数组插入排序保证稳定性，最后在根据桶的顺序依次放回原数组

- 希尔排序

  - 将待排序序列分为若干子序列（每个子序列的元素在原始数组中间距相同）；

  - 对这些子序列进行插入排序；

  - 减小每个子序列中元素之间的间距，重复上述过程直至间距减少为 ![1](data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7)。





---

2024/9/5

source: https://www.luogu.com.cn/problem/P1157

输出每个组合

```c++
int n, r;
vector<int> v;
vector<vector<int>> res;

void dfs(int start, int k)
{
    if(k > r)
    {
        res.push_back(v);
        return;
    }
    for(int i = start; i <= n ; i++)
    {
        v[k-1] = i;
        dfs(i+1, k + 1);
    }
}
```




source: https://www.luogu.com.cn/problem/P1706

输出每个排列

```c++
int n;
vector<int> v;
vector<bool> check;
void dfs(int k)
{
    if(k > n)
    {
        for(int i = 1; i <= n; i++)cout<<setw(5)<<v[i];
        cout<<endl;
        return;
    }
    for(int i = 1; i <= n; i++)
    {
        if(check[i] == false)
        {
            v[k] = i;
            check[i] = true;
            dfs(k+1);
            check[i] = false;//回溯
        }
    }
}
```

