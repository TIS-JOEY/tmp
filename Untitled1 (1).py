#!/usr/bin/env python
# coding: utf-8

# ### Dijkstra

# In[220]:


edges = [[0, 1, 4],
         [0, 6, 7],
         [1, 6, 11],
         [1, 7, 20],
         [6, 7, 1],
         [1, 2, 9],
         [2, 4, 2],
         [4, 7, 1],
         [2, 3, 6],
         [4, 3, 10],
         [7, 8, 3],
         [4, 8, 5],
         [4, 5, 15],
         [3, 5, 5],
         [8, 5, 12]]


# In[221]:


adjacency_matrix = [[float('inf') if i !=j else 0 for i in range(9)] for j in range(9)]


# In[222]:


for edge in edges:
    adjacency_matrix[edge[0]][edge[1]] = edge[2]
    adjacency_matrix[edge[1]][edge[0]] = edge[2]


# In[223]:


adjacency_matrix


# In[224]:


def dijkstra(start, adjacency_matrix):
    min_distance = [float('inf') for i in range(len(adjacency_matrix[0]))]
    min_distance[start] = 0
    queue = [start]
    travelSet = {start}
    while(len(queue)>0):
        currentNode = queue.pop()
        travelSet.add(currentNode)
        for i in range(len(adjacency_matrix[currentNode])):
            cost = adjacency_matrix[currentNode][i]
            if (min_distance[currentNode]+cost<min_distance[i]):
                min_distance[i] = min_distance[currentNode]+cost
        newNode = None
        minNewDistance = float('inf')
        for i in range(len(min_distance)):
            if i not in travelSet:
                if (newNode == None or minNewDistance>min_distance[i]):
                    newNode = i
                    minNewDistance = min_distance[i]
        
        if(newNode!=None):
            queue.append(newNode)
        print('node', currentNode, min_distance)
    print(min_distance)


# In[225]:


from collections import defaultdict
from heapq import heappush, heappop

def test(n, start, edges):
    costs = defaultdict(list)
    travelSet = {start}
    for x,y,c in edges:
        costs[x].append([y,c])
    minDist = [float('inf')]*n
    minDist[start] = 0
    queue = [[0, start]]
    while queue:
        curcost, node = heappop(queue)
        for nxt, cost in costs[node]:
            nxtcost = curcost + cost
            if (nxtcost<minDist[nxt]):
                minDist[nxt] = nxtcost
                heappush(queue, [nxtcost, nxt])
        print('node', node, minDist)


# In[226]:


edges1 = [[0,1,4],[0,6,7],[1,6,11],[1,7,20],[1,2,9],[2,3,6],[2,4,2],[3,4,10],[3,5,5],[4,5,15],[4,7,1],[4,8,5],[5,8,12],[6,7,1],[7,8,3]]
adjacency_matrix1 = [[float('inf') if i !=j else 0 for i in range(9)] for j in range(9)]
for edge in edges:
    adjacency_matrix1[edge[0]][edge[1]] = edge[2]
    adjacency_matrix1[edge[1]][edge[0]] = edge[2]


# In[227]:


dijkstra(0, adjacency_matrix1)


# ### Floyd-Warshall

# In[228]:


adjacency_matrix = [[0, 2, 6, 4],
                    [float('inf'), 0, 3, float('inf')],
                    [7, float('inf'), 0, 1],
                    [5, float('inf'), 12, 0]]


# In[229]:


def folydWarshall(adjacency_matrix):
    
    for k in range(len(adjacency_matrix[0])):
        for i in range(len(adjacency_matrix[0])):
            for j in range(len(adjacency_matrix[0])):
                adjacency_matrix[i][j] = min(adjacency_matrix[i][j], adjacency_matrix[i][k]+adjacency_matrix[k][j])
    print(adjacency_matrix)


# In[230]:


folydWarshall(adjacency_matrix)


# ### BFS

# In[231]:


class TreeNode:
    val = None
    child = []
    def __init__(self, val, child):
        self.val = val
        self.child = []
    def addChild(self, child):
        self.child.append(child)
treeNodeA = TreeNode('A', [])
treeNodeB = TreeNode('B', [])
treeNodeC = TreeNode('C', [])
treeNodeD = TreeNode('D', [])
treeNodeE = TreeNode('E', [])
treeNodeF = TreeNode('F', [])
treeNodeX = TreeNode('X', [])
treeNodeA.addChild(treeNodeB)
treeNodeA.addChild(treeNodeC)
treeNodeB.addChild(treeNodeD)
treeNodeB.addChild(treeNodeX)
treeNodeD.addChild(treeNodeE)
treeNodeC.addChild(treeNodeF)


# In[318]:


def bfs(node):
    queue = [node]
    while(len(queue)>0):
        currentNode = queue.pop(0)
        print(currentNode.val)
        for child in currentNode.child:
            queue.append(child)


# In[319]:


bfs(treeNodeA)


# ### DFS

# In[320]:


def dfs(node):
    stack = [node]
    while(len(stack)>0):
        currentNode = stack.pop()
        print(currentNode.val)
        for childIndex in range(len(currentNode.child)-1, -1, -1):
            child = currentNode.child[childIndex]
            stack.append(child)


# In[321]:


dfs(treeNodeA)


# ### Knapsap

# In[236]:


item2Value = {'A':4500,'B':5700,'C':2250,'D':1100,'E':6700}
item2Weight = {'A':4,'B':5,'C':2,'D':1,'E':6}
maxWeight = 8


# In[237]:


def knapsap(item2Value, item2Weight, maxWeight):
    dp = [0 for i in range(maxWeight+1)]
    for item, itemValue in item2Value.items():
        itemWeight = item2Weight[item]
        for currentAffordWeight in range(maxWeight+1):
            for itemCnt in range(currentAffordWeight//itemWeight):
                weightCost = itemWeight*(itemCnt+1)
                currentValue = itemValue * (itemCnt+1)
                remainWeight = currentAffordWeight - weightCost
                if (remainWeight >= 0):
                    dp[currentAffordWeight] = max(dp[currentAffordWeight], dp[remainWeight]+currentValue)
    print('dp', dp)


# In[238]:


knapsap(item2Value, item2Weight, maxWeight)


# ### Bellman-Ford

# In[322]:


edges = [[1, 2, 2], [0, 1, -3], [0, 4, 5], [3, 4, 2], [2, 3, 3]]
start = 0


# In[323]:


def bellmanFord(edges, totalLen, start):
    min_distance = [float('inf') if i!=start else 0 for i in range(totalLen)]
    withoutRenew = False
    while(withoutRenew == False):
        withoutRenew = True
        for edge in edges:
            startPoint = edge[0]
            destPoint = edge[1]
            cost = edge[2]
            if (min_distance[destPoint]>min_distance[startPoint]+cost):
                withoutRenew = False
                min_distance[destPoint] = min_distance[startPoint]+cost
    return min_distance


# In[324]:


min_distance = bellmanFord(edges, 5, 0)
print(min_distance)


# #### check has cycle

# In[477]:


edges = [[0, 1], [1, 2], [2, 3], [0, 4], [3, 4]]


# In[478]:


def checkHasCycle(edges, maxN):
    group2Members = {i:{i} for i in range(maxN)}
    for edge in edges:
        
        for group, members in group2Members.items():
            if (edge[0] in members):
                group2Members[group] = members.union(group2Members[edge[1]])
        if (edge[0] in group2Members[edge[1]]):
            return True
    return False
        


# In[479]:


checkHasCycle(edges, 5)


# ### truck

# In[249]:


truckSize = 5
unitsPerBox = [3, 2, 1]
boxes = [1, 2, 3]


# In[250]:


dp = [0 for i in range(truckSize+1)]


# In[251]:


for currentBoxIndex in range(len(boxes)):
    currentBoxUnit = unitsPerBox[currentBoxIndex]
    affordSize2Units = {}
    for affordSize in range(len(dp)):
        for boxCnt in range(1, boxes[currentBoxIndex]+1):
            units = currentBoxUnit * boxCnt
            remainAffordSize = affordSize - boxCnt
            if (remainAffordSize >= 0):
                renewValue = max(dp[affordSize], units + dp[remainAffordSize])
                if (affordSize in affordSize2Units):
                    affordSize2Units[affordSize] = max(affordSize2Units[affordSize], renewValue)
                else:
                    affordSize2Units[affordSize] = renewValue
    for affordSize, units in affordSize2Units.items():
        dp[affordSize] = units
    print(dp)


# ### Disjoint set

# In[263]:


connections = [(0, 4), (3, 5), (1, 2), (2, 5), (5, 1)]
groups = [i for i in range(6)]
groups


# In[266]:


def findDisjointSet(connections, groups):
    for connection in connections:
        p = min(connection)
        q = max(connection)
        for i in range(len(groups)):
            if (groups[i] == groups[q]):
                groups[i] = groups[p]
    return groups


# In[267]:


findDisjointSet(connections, groups)


# In[486]:


connections = [(0, 4), (3, 5), (1, 2), (2, 5), (5, 1)]
def findDisjointSet1(connections, maxN):
    group2Members = {i:{i} for i in range(maxN)}
    for connection in connections:
        
        for group, members in group2Members.items():
            if (connection[0] in members):
                group2Members[group] = members.union(group2Members[connection[1]])

    return group2Members
        


# In[487]:


findDisjointSet1(connections, 6)


# ### Dam

# In[498]:


wallPosition = [0, 7]
wallHeights = [2, 4]


# In[505]:


def damDeisgn(wallPosition, wallHeights):
    result = 0
    for i in range(len(wallPosition)-1):
        heightDiff = abs(wallHeights[i+1] - wallHeights[i])
        gapDiff = abs(wallPosition[i+1] - wallPosition[i] - 1)
        localMax = 0
        if (gapDiff > heightDiff):
            #5
            low = max(wallHeights[i+1], wallHeights[i])+1
            remGap = gapDiff - heightDiff - 1
            localMax = low + remGap // 2
        else:
            localMax = min(wallHeights[i+1], wallHeights[i])+gapDiff
        result = max(result, localMax)
    return result


# In[506]:


damDeisgn(wallPosition, wallHeights)


# ### Keyboard

# In[546]:


def generatekeyboardAdjencyMatrix(n):
    import math
    adjency_matrix = [[0]*n for i in range(n)]
    col = int(math.sqrt(n))
    
    for i in range(n):
        baseAxis = (i//col, i%col)
        for j in range(n):
            travelAxis = (j//col, j%col)
            diff = (abs(baseAxis[0]-travelAxis[0]), abs(baseAxis[1]-travelAxis[1]))
            adjency_matrix[i][j] = max(diff)
    return adjency_matrix


# In[558]:


def entryTime(s, keypad):
    adjency_matrix = generatekeyboardAdjencyMatrix(len(keypad))
    key2Index = {keypad[i]:i for i in range(len(keypad))}
    result = 0
    currentPos = key2Index[s[0]]
    for i in s:
        result+=(adjency_matrix[currentPos][key2Index[i]])
        currentPos = key2Index[i]
    return result


# In[559]:


entryTime('423692','923857614')


# ### Connected groups

# In[563]:


def findDisjoinSet(connections, groups):
    for connection in connections:
        p = min(connection)
        q = max(connection)
        for i in range(len(groups)):
            if (groups[i] == groups[q]):
                groups[i] = groups[p]
    return groups


# In[564]:


def countGroups(related):
    connections = []
    for i in range(len(related)):
        for j in range(len(related[i])):
            if (related[i][j] == '1' and i!=j):
                connections.append([i, j])
    groups = [i for i in range(len(related[0]))]
    return len(set(findDisjoinSet(connections, groups)))


# In[565]:


countGroups(['1100','1110','0110','0001'])


# ### Highly Profitable Months

# In[578]:


def findLastZeroIndex(target):
    for i in range(len(target)-1, -1, -1):
        if (target[i]=='0'):
            return i
    return -1
def countHighlyProfitableMonths(stockPrices, k):
    result = 0
    recordList = []
    start1Index = None
    if (len(stockPrices)<k):
        return 0
    if (len(stockPrices)==1):
        if (k==1):
            return 1
        else:
            0
    for i in range(len(stockPrices)-1):
        if (stockPrices[i+1]>stockPrices[i]):
            if (start1Index == None):
                start1Index = i
        else:
            if (start1Index != None):
                recordList.append((start1Index, i))
                start1Index = None

    if (stockPrices[-1]>stockPrices[-2]):
        recordList.append((start1Index, len(stockPrices)-1))
    for record in recordList:
        if (record[1]-record[0]-k+2)>0:
            result+=(record[1]-record[0]-k+2)
    return result


# In[579]:


countHighlyProfitableMonths([1, 2, 3, 4], 4)


# ### kruskal

# In[621]:


edges = [(0, 1, 4), (0, 7, 8), (1, 7, 11), (1, 2, 8), (7, 8, 7), (7, 6, 1), (2, 8, 2), (8, 6, 6), (6, 5, 2), (2, 5, 4), (2, 3, 7), (3, 5, 14), (3, 4, 9), (4, 5, 10)]


# In[622]:


def kruskal(edges, totalPointCnt):
    sorted_edges = sorted(edges, key=lambda tup: tup[2])
    edgeCnt = 0
    result = []
    groups = [i for i in range(totalPointCnt)]
    while(edgeCnt<totalPointCnt and len(sorted_edges)>0):
        currentEdge = sorted_edges.pop(0)

        p = min([currentEdge[0], currentEdge[1]])
        q = max([currentEdge[0], currentEdge[1]])
        print('add edges', currentEdge)
        if (groups[p] == groups[q]):
            continue
        else:
            for i in range(len(groups)):
                if (groups[i] == groups[q] and i != q):
                    groups[i] = groups[p]
            groups[q] = groups[p]
            edgeCnt+=1
            result.append(currentEdge)
    return result


# In[623]:


kruskal(edges, 9)


# ### bubble sort

# In[624]:


def bubble_sort(arr):
    def swap(x, y):
        arr[x], arr[y] = arr[y], arr[x]
 
    iter_time = len(arr)-1
    for i in range(iter_time):
        for j in range(0, iter_time-i):
            if arr[j] > arr[j+1]:
                swap(j, j+1)


# ### Selection Sort

# In[625]:


def selection_sort(arr):
    def swap(x, y):
        arr[x], arr[y] = arr[y], arr[x]
 
    for i in range(len(arr) - 1):
 
        index = i
 
        for j in range(i+1, len(arr)):
            if arr[j] < arr[index]:
                index = j
 
        if index != i:
            swap(index, i)
 
    return arr


# ### Insertion Sort

# In[626]:


def insertion_sort(arr):   
    def swap(x, y):
        arr[x], arr[y] = arr[y], arr[x]
 
    for i in range(len(arr)):
        j = i
        while j > 0 and arr[j-1] > arr[j]:
            swap(j, j-1)
            j -= 1
 
    return arr


# ### quick sort

# In[627]:


def quick_sort(arr):
    arr = quick_sort_recur(arr, 0, len(arr)-1)
    return arr
 
def quick_sort_recur(arr, first, last):
    if first < last:
        flag = partition(arr, first, last)
        # Start our two recursive calls
        quick_sort_recur(arr, first, flag-1)
        quick_sort_recur(arr, flag+1, last)
 
    return arr
 
def partition(arr, first, last):
    def swap(x, y):
        if x != y:
            arr[x], arr[y] = arr[y], arr[x]
 
    flag = first
 
    for i in range(first, last):
        if arr[i] < arr[last]:  # last is the pivot
            swap(flag, i)
            flag += 1
 
    swap(flag, last)
 
    return flag


# ### merge sort

# In[628]:


def merge_sort(arr):
    def merge(left, right, merged):
        lc, rc = 0, 0  # left cursor, right cursor
 
        while lc < len(left) and rc < len(right):
            if left[lc] <= right[rc]:
                merged[lc+rc] = left[lc]
                lc += 1
            else:
                merged[lc+rc] = right[rc]
                rc += 1
 
        # Add left overs
        for i in range(lc, len(left)):
            merged[i+rc] = left[i]
        for i in range(lc, len(right)):
            merged[lc+i] = right[i]
 
        return merged
 
    if len(arr) <= 1:
        return arr
 
    # divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
 
    # conquer
    return merge(left, right, arr.copy())


# ### topoSort

# In[631]:


def topoSort(graph):     
    in_degrees = dict((u,0) for u in graph)   #初始化所有顶点入度为0     
    num = len(in_degrees)     
    for u in graph:         
        for v in graph[u]:             
            in_degrees[v] += 1    #计算每个顶点的入度
    print('in_degrees',in_degrees)
    Q = [u for u in in_degrees if in_degrees[u] == 0]   # 筛选入度为0的顶点     
    Seq = []     
    while Q:         
        u = Q.pop()       #默认从最后一个删除         
        Seq.append(u)         
        for v in graph[u]:             
            in_degrees[v] -= 1    #移除其所有出边
            if in_degrees[v] == 0:        
                Q.append(v)          #再次筛选入度为0的顶点
    if len(Seq) == num:       #输出的顶点数是否与图中的顶点数相等
        return Seq     
    else:         
        return None

G = {
    'a':'bf',
    'b':'cdf',
    'c':'d',
    'd':'ef',
    'e':'f',
    'f':''
}
print(topoSort(G))


# In[ ]:





# In[ ]:




