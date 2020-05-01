class Solution:
    # https://leetcode.com/problems/two-sum
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict = {}
        for i in range(len(nums)):
            if (target - nums[i]) in dict:
                return [dict[target - nums[i]], i]
            dict[nums[i]] = i

    # https://leetcode.com/problems/add-two-numbers
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        adder = 0
        dummy = ListNode(-1)
        prev = dummy
        while l1 or l2:
            if l1 is not None:
                adder += l1.val
                l1 = l1.next
            
            if l2 is not None:
                adder += l2.val
                l2 = l2.next
                
            prev.next = ListNode(adder % 10)
            prev = prev.next
            adder //= 10
        
        if adder > 0:
            prev.next = ListNode(adder)
        
        return dummy.next

    # https://leetcode.com/problems/valid-parenthesis-string
    def checkValidString(self, s: str) -> bool:
        low, high = 0, 0
        for ch in s:
            low += 1 if ch is '(' else -1
            high += -1 if ch is ')' else 1
            if (high < 0):
                return False
            low = max(low, 0)
        return low is 0

    # https://leetcode.com/problems/remove-element/
    def removeElement(self, nums: List[int], val: int) -> int:
        len = 0
        for num in nums:
            if num is not val:
                nums[len], len = num, len + 1
        return len

    # https://leetcode.com/problems/substring-with-concatenation-of-all-words/
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not s or not words or not words[0]:
            return []
        
        freq = {}
        for word in words:
            freq[word] = freq[word] + 1 if word in freq else 1
        
        result = []
        k = len(words[0])
        n = len(s)
        for i in range(k):
            start = i
            wordCount = len(words)
            temp = {}
            
            for j in range(i, n, k):
                sub = s[j : j + k]
                if sub in freq:
                    temp[sub] = temp[sub] + 1 if sub in temp else 1
                    wordCount -= 1
                    
                    while temp[sub] > freq[sub]:
                        str = s[start : start + k]
                        start += k
                        temp[str] -= 1
                        wordCount += 1
                    
                    if wordCount == 0:
                        result.append(start)
                else:
                    temp = {}
                    wordCount = len(words)
                    start = j + k
        return result

    # https://leetcode.com/problems/search-insert-position/
    def searchInsert(self, nums: List[int], target: int) -> int:
        if not nums:
            return 0
        
        if nums[-1] < target:
            return len(nums)
        
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if (nums[mid] < target):
                start = mid
            else:
                end = mid
        
        if nums[start] >= target:
            return start
        return end

    # https://leetcode.com/problems/length-of-last-word/
    def lengthOfLastWord(self, s: str) -> int:
        if not s:
            return 0
        
        n, last = len(s), -1
        while last >= -n and s[last] == ' ':
            last -= 1
        
        i = last
        while i >= -n and s[i] != ' ':
            i -= 1
        
        return last - i

    # https://leetcode.com/problems/spiral-matrix-ii/
    def generateMatrix(self, n: int) -> List[List[int]]:
        if n <= 0:
            return []
        
        matrix = [[0] * n for _ in range(n)]
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        
        x, y, d = 0, 0, 0
        for num in range(n * n):
            matrix[x][y] = num + 1
            x1, y1 = x + dx[d], y + dy[d]
            if x1 < 0 or x1 >= n or y1 < 0 or y1 >= n or matrix[x1][y1] != 0:
                d = (d + 1) % 4
                x1, y1 = x + dx[d], y + dy[d]
            x, y = x1, y1
        
        return matrix

    # https://leetcode.com/problems/rotate-list/
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if k == 0 or not head:
            return head
        
        p, n = head, 0
        while p:
            n, p = n + 1, p.next
            
        k %= n
        if k == 0:
            return head
        
        p, q = head, head
        for i in range(k):
            p = p.next
        
        while p.next:
            p, q = p.next, q.next
        
        ans, q.next, p.next = q.next, None, head
        return ans

    # https://leetcode.com/problems/minimum-path-sum
    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid or len(grid) == 0 or not grid[0] or len(grid[0]) == 0:
            return 0
        
        m, n = len(grid), len(grid[0])
        
        for i in range(1, n):
            grid[0][i] += grid[0][i - 1]
            
        for i in range(1, m):
            grid[i][0] += grid[i - 1][0]
        
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        
        return grid[-1][-1]

    # https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums or len(nums) == 0:
            return 0
        
        n = 0
        for num in nums:
            if n < 2 or nums[n - 2] < num:
                nums[n] = num
                n += 1
        return n

    # https://leetcode.com/problems/text-justification/
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        ans, n, start, i = [], len(words), 0, 0
        
        while i < n:
            count = 0
            while i < n and count + len(words[i]) + i - start <= maxWidth:
                count += len(words[i])
                i += 1
            
            total = maxWidth - count
            spaces = 1 if i == n or start + 1 == i else total // (i - 1 - start)
            total = 0 if i == n else total - (i - 1 - start) * spaces
            
            line = [words[start]]
            for j in range(start + 1, i):
                if total > 0:
                    line.append(' ' * (spaces + 1))
                    total -= 1
                else:
                    line.append(' ' * spaces)
                line.append(words[j])
            
            temp = ''.join(line)
            temp += ' ' * (maxWidth - len(temp))
            ans.append(temp)
            start = i
        return ans

    # https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = ListNode(-1)
        prev = dummy
        while head:
            if not head.next or head.next.val > head.val:
                prev.next, prev = head, head
            while head.next and head.next.val == head.val:
                head = head.next
            head = head.next
        prev.next = None
        return dummy.next

    # https://leetcode.com/problems/remove-duplicates-from-sorted-list/submissions/
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        p = head
        while p:
            while p.next and p.next.val == p.val:
                p.next = p.next.next
            p = p.next
        return head

    # https://leetcode.com/problems/partition-list/
    def partition(self, head: ListNode, x: int) -> ListNode:
        left = prevLeft = ListNode(-1)
        right = prevRight = ListNode(-1)
        while head:
            if head.val < x:
                prevLeft.next, prevLeft = head, head
            else:
                prevRight.next, prevRight = head, head
            head = head.next
        prevRight.next = None
        prevLeft.next = right.next
        return left.next

    # https://leetcode.com/problems/scramble-string/
    @lru_cache(None)
    def isScramble(self, s1: str, s2: str) -> bool:
        if len(s1) != len(s2):
            return False
        if s1 == s2:
            return True
        if sorted(s1) != sorted(s2):
            return False
        
        for i in range(1, len(s1)):
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]) or \
               self.isScramble(s1[:i], s2[len(s1)-i:]) and self.isScramble(s1[i:], s2[:len(s1)-i]):
                return True
        return False

    # https://leetcode.com/problems/gray-code/
    def grayCode(self, n: int) -> List[int]:
        result = [0]
        for i in range(n):
            result += [x + (1 << i) for x in reversed(result)]
        return result

    # https://leetcode.com/problems/unique-binary-search-trees-ii/
    def generateTrees(self, n: int) -> List[TreeNode]:
        if n < 1:
            return []
        
        @lru_cache(None)
        def generate(first, last):
            result = []
            for root in range(first, last + 1):
                for left in generate(first, root - 1):
                    for right in generate(root + 1, last):
                        node = TreeNode(root)
                        node.left, node.right = left, right
                        result.append(node)
            return result or [None]
        return generate(1, n)

    # https://leetcode.com/problems/interleaving-string/
    @lru_cache(None)
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        
        if not len(s1):
            return s2 == s3
        
        if not len(s2):
            return s1 == s3
        
        return s1[-1] == s3[-1] and self.isInterleave(s1[:-1], s2, s3[:-1]) or \
               s2[-1] == s3[-1] and self.isInterleave(s1, s2[:-1], s3[:-1])

    # https://leetcode.com/problems/same-tree/
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:        
        if p and q:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return p is q

    # https://leetcode.com/problems/recover-binary-search-tree/
    def recoverTree(self, root: TreeNode) -> None:
        first, second, prev, curr = None, None, TreeNode(float('-inf')), root
        while curr:
            if curr.left:
                temp = curr.left
                while temp.right and temp.right != curr: temp = temp.right
                if not temp.right:
                    temp.right, curr = curr, curr.left
                    continue
                temp.right = None
            if prev.val > curr.val:
                if not first: first = prev
                second = curr
            prev, curr = curr, curr.right
        first.val, second.val = second.val, first.val

    # https://leetcode.com/problems/subarray-sum-equals-k/
    def subarraySum(self, nums: List[int], k: int) -> int:
        dict, sum, ans = {0 : 1}, 0, 0
        for num in nums:
            sum += num
            ans += dict.get(sum - k, 0)
            dict[sum] = dict.get(sum, 0) + 1
        return ans

    # https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        if not inorder or not postorder or len(inorder) != len(postorder):
            return None
        dict = {}
        for i in range(len(inorder)):
            dict[inorder[i]] = i
        
        def helper(inorder, postorder, dict, iStart, iEnd, pStart, pEnd):
            if iStart > iEnd or pStart > pEnd: return None
            node, index = TreeNode(postorder[pEnd]), dict[postorder[pEnd]]
            count = index - iStart
            node.left, node.right = helper(inorder, postorder, dict, iStart, index - 1, pStart, pStart + count - 1), helper(inorder, postorder, dict, index + 1, iEnd, pStart + count, pEnd - 1)
            return node
        return helper(inorder, postorder, dict, 0, len(inorder) - 1, 0, len(postorder) - 1)

    # https://leetcode.com/problems/distinct-subsequences/
    @lru_cache(None)
    def numDistinct(self, s: str, t: str) -> int:
        if len(s) < len(t):
            return 0
        n1, n2 = len(s), len(t)
        f = [1] + [0] * n2
        for i in range(n1):
            upper = min(i, n2 - 1)
            for j in range(upper, -1, -1):
                if s[i] == t[j]:
                    f[j + 1] += f[j]
        return f[n2]

    # https://leetcode.com/problems/pascals-triangle-ii/
    def getRow(self, rowIndex: int) -> List[int]:
        row = [1]
        for i in range(rowIndex):
            temp = [1]
            for j in range(i):
                temp.append(row[j] + row[j + 1])
            temp.append(1)
            row = temp
        return row

    # https://leetcode.com/problems/triangle/
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle:
            return float('inf')
        last, n = triangle[0], len(triangle)
        for i in range(1, n):
            triangle[i][0] += last[0]
            for j in range(1, i):
                triangle[i][j] += min(last[j - 1], last[j])
            triangle[i][i] += last[i - 1]
            last = triangle[i]
        
        ans = float('inf')
        for i in range(n):
            ans = min(ans, triangle[n - 1][i])
        return ans

    # https://leetcode.com/problems/bitwise-and-of-numbers-range/
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        count = 0
        while m != n:
            m >>= 1
            n >>= 1
            count += 1
        return m << count

    # https://leetcode.com/problems/palindrome-partitioning-ii/
    def minCut(self, s: str) -> int:
        if not s:
            return 0
        
        n = len(s)
        f = [x - 1 for x in range(n + 1)]
        for mid in range(n):
            start, end = mid, mid
            while start >= 0 and end < n and s[start] == s[end]:
                f[end + 1] = min(f[end + 1], f[start] + 1)
                start, end = start - 1, end + 1
                
            start, end = mid, mid + 1
            while start >= 0 and end < n and s[start] == s[end]:
                f[end + 1] = min(f[end + 1], f[start] + 1)
                start, end = start - 1, end + 1
        return f[n]

    # https://leetcode.com/problems/candy/
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        ans, count, pre = 1, 0, 1
        for i in range(1, n):
            if ratings[i] >= ratings[i - 1]: # non-descending case
                if count > 0: # position (i - 1) is turning point for \/
                    ans += (count + 1) * count // 2 # total (count + 1) in descending slope, add sum(1,2,..,count) 
                    if pre <= count: # pre is turning point for /\
                        ans += count - pre + 1 # add candy to ensure turing point is max in both / and \
                    count, pre = 0, 1
                pre = 1 if ratings[i] == ratings[i - 1] else pre + 1 # give 1 candy when having the same rating
                ans += pre # add candy for position i (second in increasing slope)
            else:
                count += 1 # one more in descending slope
        if count > 0:
            ans += (count + 1) * count // 2
            if pre <= count:
                ans += count - pre + 1
        return ans

    # https://leetcode.com/problems/insertion-sort-list/
    def insertionSortList(self, head: ListNode) -> ListNode:
        dummy = pre = ListNode(-1)
        while head:
            if head.val < pre.val:
                pre = dummy
            p = pre.next
            while p and p.val < head.val:
                pre, p = p, p.next
            p, head.next, pre.next, pre = head.next, pre.next, head, head
            head = p
        return dummy.next

    # https://leetcode.com/problems/jump-game
    def canJump(self, nums: List[int]) -> bool:
        curr, n, maxJump = 0, len(nums), 0
        while curr <= maxJump and maxJump < n - 1:
            maxJump = max(maxJump, curr + nums[curr])
            curr += 1
        return maxJump >= n - 1

    # https://leetcode.com/problems/diagonal-traverse-ii/
    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        ans = []
        for i, row in enumerate(nums):
            for j, num in enumerate(row):
                if len(ans) <= i + j:
                    ans.append([])
                ans[i + j].append(num)
        return [x for row in ans for x in reversed(row)]

    # https://leetcode.com/problems/longest-common-subsequence/
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n1, n2 = len(text1), len(text2)
        f = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        for i in range(n1):
            for j in range(n2):
                if text1[i] == text2[j]:
                    f[i + 1][j + 1] = f[i][j] + 1
                else:
                    f[i + 1][j + 1] = max(f[i + 1][j], f[i][j + 1])
        return f[n1][n2]    

    # https://leetcode.com/problems/maximum-gap/
    def maximumGap(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2: return 0
        
        a, b = min(nums), max(nums)
        if a == b: return 0
        
        diff = ceil((b - a) / (n - 1))
        buckets = [[a, a]] + [[1 << 32, -1] for _ in range(n - 3)] + [[b, b]]
        for x in nums:
            if x == a or x == b:
                continue
            bucket = buckets[(x - a) // diff]
            bucket[0], bucket[1] = min(bucket[0], x), max(bucket[1], x)
        
        ans, prev = diff, buckets[0][1]
        for i in range(1, n - 1):
            if buckets[i][0] < 1 << 32:
                ans = max(ans, buckets[i][0] - prev)
                prev = buckets[i][1]
        return ans

    # https://leetcode.com/problems/maximal-square
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        m, n, ans = len(matrix), len(matrix[0]), 0
        f = [[0] * n for _ in range(m)]
        for i in range(n):
            if matrix[0][i] == '1':
                f[0][i] = 1
                ans = 1
        for i in range(1, m):
            if matrix[i][0] == '1':
                f[i][0] = 1
                ans = 1
        
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == '1':
                    f[i][j] = min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1]) + 1
                    ans = max(ans, f[i][j])
        return ans * ans

    # https://leetcode.com/problems/excel-sheet-column-title/
    def convertToTitle(self, n: int) -> str:
        # 26-nary, ((x[0]*26 + x[1])*26 + x[2])*26 + ...
        return self.convertToTitle((n - 1) // 26) + chr((n - 1) % 26 + ord('A')) if n > 0 else ''

    # https://leetcode.com/problems/compare-version-numbers/
    def compareVersion(self, version1: str, version2: str) -> int:
        def getLevels(s):
            s = list(map(int, s.split('.')))
            i = len(s) - 1
            while i >= 0 and s[i] == 0:
                i -= 1
            return s[:i+1]
        levels1, levels2 = getLevels(version1), getLevels(version2)
        return 1 if levels1 > levels2 else (-1 if levels1 < levels2 else 0)

    # https://leetcode.com/problems/create-maximum-number/
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        def maxArray(nums, k):
            n, stack = len(nums), []
            for i, num in enumerate(nums):
                while stack and stack[-1] < num and n - i + len(stack) > k:
                    stack.pop(-1)
                if len(stack) < k:
                    stack.append(num)
            return stack
        
        def merge(a, b):
            return [max(a, b).pop(0) for _ in a + b]
        
        return max(merge(maxArray(nums1, i), maxArray(nums2, k - i))
                   for i in range(k + 1)
                   if i <= len(nums1) and k - i <= len(nums2))

    # https://leetcode.com/problems/dungeon-game/
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        if not dungeon:
            return 1
        m, n = len(dungeon), len(dungeon[0])
        f = [[0] * n for _ in range(m)]
        f[m - 1][n - 1] = max(1 - dungeon[m - 1][n - 1], 1)
        
        for i in range(n - 2, -1, -1):
            f[m - 1][i] = max(f[m - 1][i + 1] - dungeon[m - 1][i], 1)
        for i in range(m - 2, -1, -1):
            f[i][n - 1] = max(f[i + 1][n - 1] - dungeon[i][n - 1], 1)
            
        for i in range(m - 2, -1, -1):
            for j in range(n - 2, -1, -1):
                f[i][j] = max(min(f[i + 1][j], f[i][j + 1]) - dungeon[i][j], 1)
        return f[0][0]

    # https://leetcode.com/problems/integer-break/
    def integerBreak(self, n: int) -> int:
        @lru_cache(None)
        def helper(n, split):
            if n == 1: return 1
            ans = 1 if not split else n
            for i in range(1, n // 2 + 1):
                ans = max(ans, helper(i, True) * helper(n - i, True))
            return ans
        return helper(n, False)
