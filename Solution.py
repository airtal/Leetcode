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

    # https://leetcode.com/problems/jewels-and-stones/
    def numJewelsInStones(self, J: str, S: str) -> int:
        jewels, ans = set(), 0
        for l in J:
            if l not in jewels:
                jewels.add(l)
        for l in S:
            if l in jewels:
                ans += 1
        return ans

    # https://leetcode.com/problems/ransom-note/
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        count = {}
        for l in magazine:
            count[l] = count.get(l, 0) + 1
        for l in ransomNote:
            if count.get(l, 0) == 0:
                return False
            count[l] -= 1
        return True

    # https://leetcode.com/problems/destination-city/
    def destCity(self, paths: List[List[str]]) -> str:
        cities, out = set(), set()
        for path in paths:
            cities.add(path[0])
            cities.add(path[1])
            if path[0] not in out:
                out.add(path[0])
        return (cities - out).pop()

    # https://leetcode.com/problems/check-if-all-1s-are-at-least-length-k-places-away/
    def kLengthApart(self, nums: List[int], k: int) -> bool:
        last = -k - 1
        for i, num in enumerate(nums):
            if num == 1:
                if i - last - 1 < k:
                    return False
                last = i
        return True
    
    # https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        mind, maxd, start = collections.deque(), collections.deque(), 0
        for i, a in enumerate(nums):
            while len(mind) and a <= nums[mind[-1]]:
                mind.pop()
            while len(maxd) and a >= nums[maxd[-1]]:
                maxd.pop()
            mind.append(i)
            maxd.append(i)
            if nums[maxd[0]] - nums[mind[0]] > limit:
                if mind[0] == start:
                    mind.popleft()
                if maxd[0] == start:
                    maxd.popleft()
                start += 1
        return len(nums) - start

    # https://leetcode.com/problems/number-complement/
    def findComplement(self, num: int) -> int:
        n = 1
        while n < num:
            n = (n << 1) | 1
        return n - num

    # https://leetcode.com/problems/first-unique-character-in-a-string/
    def firstUniqChar(self, s: str) -> int:
        count, start = {}, 0
        for i, l in enumerate(s):
            count[l] = count.get(l, 0) + 1
            while start <= i and count[s[start]] > 1:
                start += 1
        return start if start < len(s) else -1

    # https://leetcode.com/problems/majority-element
    def majorityElement(self, nums: List[int]) -> int:
        ans, cnt = -1, 0
        for num in nums:
            if cnt == 0:
                ans, cnt = num, 1
            elif num == ans:
                cnt += 1
            else:
                cnt -= 1
        return ans

    # https://leetcode.com/problems/cousins-in-binary-tree/
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        if not root or x == root.val or y == root.val: return False
        q = deque([root])
        while q:
            valSet, nextq = set(), deque()
            for node in q:
                if node.left:
                    nextq.append(node.left)
                    valSet.add(node.left.val)
                if node.right:
                    nextq.append(node.right)
                    valSet.add(node.right.val)
                if node.left and node.right:
                    if nextq[-1].val == x and nextq[-2].val == y or \
                       nextq[-1].val == y and nextq[-2].val == x:
                        return False
            if x in valSet and y in valSet:
                return True
            q = nextq
        return False

    # https://leetcode.com/problems/check-if-it-is-a-straight-line/
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        x, y = coordinates[0][0], coordinates[0][1]
        deltaX, deltaY = coordinates[1][0] - x, coordinates[1][1] - y
        for i in range(2, len(coordinates)):
            if (coordinates[i][1] - y) * deltaX != (coordinates[i][0] - x) * deltaY: return False
        return True

    # https://leetcode.com/problems/valid-perfect-square/
    def isPerfectSquare(self, num: int) -> bool:
        start, end = 1, num
        while start + 1 < end:
            mid = start + (end - start) // 2
            if mid ** 2 > num:
                end = mid
            else:
                start = mid
        if start ** 2 == num or end ** 2 == num:
            return True
        return False

    # https://leetcode.com/problems/find-the-town-judge/
    def findJudge(self, N: int, trust: List[List[int]]) -> int:
        degree = [0] * (N + 1)
        for p1, p2 in trust:
            degree[p2] += 1
            degree[p1] -= 1
        
        for i in range(1, N + 1):
            if degree[i] == N - 1:
                return i
        return -1

    # https://leetcode.com/problems/flood-fill
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        if image[sr][sc] == newColor: return image
        m, n = len(image), len(image[0])
        
        def helper(r, c):
            color, image[r][c] = image[r][c], newColor
            for dx, dy in [[0,1], [1,0], [0,-1], [-1,0]]:
                x, y = r + dx, c + dy
                if x < 0 or x >= m or y < 0 or y >= n or image[x][y] != color: continue
                helper(x, y)
        helper(sr, sc)
        return image

    # https://leetcode.com/problems/build-an-array-with-stack-operations/
    def buildArray(self, target: List[int], n: int) -> List[str]:
        ans, prev = [], 0
        if not target: return ans
        
        for curr in target:
            ans += ['Push', 'Pop'] * (curr - prev - 1) + ['Push']
            prev = curr
        return ans

    # https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/
    def ways(self, pizza: List[str], k: int) -> int:
        m, n = len(pizza), len(pizza[0])
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                f[i + 1][j + 1] = f[i + 1][j] + f[i][j + 1] - f[i][j]
                if pizza[i][j] == 'A':
                    f[i + 1][j + 1] += 1
        
        @lru_cache(None)
        def helper(r1, r2, c1, c2, k):
            #print(r1, r2, c1, c2, k)
            if k == 1: return 1
            ans, cnt = 0, f[r2][c2] - f[r2][c1] - f[r1][c2] + f[r1][c1]
            for i in range(r1 + 1, r2):
                up = f[i][c2] - f[i][c1] - f[r1][c2] + f[r1][c1]
                if up > 0 and cnt - up > 0:
                    ans = (ans + helper(i, r2, c1, c2, k - 1)) % 1000000007
            for i in range(c1 + 1, c2):
                left = f[r2][i] - f[r2][c1] - f[r1][i] + f[r1][c1]
                if left > 0 and cnt - left > 0:
                    ans = (ans + helper(r1, r2, i, c2, k - 1)) % 1000000007
            return ans
        return helper(0, m, 0, n, k)

    # https://leetcode.com/problems/minimum-time-to-collect-all-apples-in-a-tree/
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        app = set()
        for i in range(n):
            if hasApple[i]:
                app.add(i)
        
        adjList = {}
        for u, v in edges:
            e = adjList.get(u, [])
            e.append(v)
            adjList[u] = e
            e = adjList.get(v, [])
            e.append(u)
            adjList[v] = e
        
        def helper(parent, curr):
            ans = 0
            for v in adjList[curr]:
                if v != parent:
                    cnt = helper(curr, v)
                    if cnt > 0 or v in app:
                        ans += cnt + 1
            return ans
        return helper(-1, 0) * 2

    # https://leetcode.com/problems/single-element-in-a-sorted-array
    def singleNonDuplicate(self, nums: List[int]) -> int:
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            cnt = mid - start + 1
            if cnt % 2 == 0:
                if nums[mid] == nums[mid + 1]:
                    end = mid - 1
                else:
                    start = mid + 1
            else:
                if nums[mid] == nums[mid + 1]:
                    start = mid
                else:
                    end = mid
        if start == 0 or nums[start] != nums[start - 1]:
            return nums[start]
        return nums[end]

    # https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/
    def countTriplets(self, arr: List[int]) -> int:
        ans, curr = 0, 0
        count = {0 : [1, 0]}
        for i, num in enumerate(arr):
            curr ^= num
            n, total = count.get(curr, [0, 0])
            # preXor[n] indicates first n elements, preXOR[k+1] == preXOR[i], j in range(i + 1, k)
            ans += i * n - total
            count[curr] = [n + 1, total + i + 1]
        return ans

    # https://leetcode.com/problems/remove-k-digits/
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        for d in num:
            while stack and k and d < stack[-1]:
                k -= 1
                stack.pop()
            stack.append(d)
            
        while stack and k:
            stack.pop()
            k -= 1
        return ''.join(stack).lstrip('0') or '0'

    # https://leetcode.com/problems/implement-trie-prefix-tree/
class TrieNode:
    def __init__(self):
        self.isWord = False
        self.children = {}
            
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for l in word:
            if l not in node.children:
                node.children[l] = TrieNode()
            node = node.children[l]
        node.isWord = True

    def search(self, word: str) -> bool:
        node = self.root
        for l in word:
            if l not in node.children:
                return False
            node = node.children[l]
        return node.isWord

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for l in prefix:
            if l not in node.children:
                return False
            node = node.children[l]
        return True

    # https://leetcode.com/problems/maximum-sum-circular-subarray/
    def maxSubarraySumCircular(self, A: List[int]) -> int:
        minAns, maxAns, minSum, maxSum, curr = 0, -float('inf'), 0, -float('inf'), 0
        for a in A:
            curr += a
            maxAns = max(maxAns, curr - minSum)
            minAns = min(minAns, curr - maxSum)
            minSum = min(minSum, curr)
            maxSum = max(maxSum, curr)
        return max(maxAns, curr - minAns)

    # https://leetcode.com/problems/odd-even-linked-list
    def oddEvenList(self, head: ListNode) -> ListNode:
        odd, even, l = ListNode(), ListNode(), 1
        oddPrev, evenPrev = odd, even
        while head:
            if l % 2 == 1:
                oddPrev.next = head
                oddPrev = head
            else:
                evenPrev.next = head
                evenPrev = head
            head, l = head.next, l + 1
        oddPrev.next, evenPrev.next = even.next, None
        return odd.next

    # https://leetcode.com/problems/find-all-anagrams-in-a-string/
    def findAnagrams(self, s: str, p: str) -> List[int]:
        freq, temp = {}, {}
        for ch in p:
            freq[ch] = freq.get(ch, 0) + 1
        
        ans, n, start = [], len(p), 0
        for i, ch in enumerate(s):
            temp[ch] = temp.get(ch, 0) + 1
            if i >= n - 1:
                if temp == freq:
                    ans += [start]
                if temp[s[start]] == 1:
                    temp.pop(s[start])
                else:
                    temp[s[start]] -= 1
                start += 1
        return ans

    # https://leetcode.com/problems/permutation-in-string
    def checkInclusion(self, s1: str, s2: str) -> bool:
        freq, temp, n = {}, {}, len(s1)
        for ch in s1:
            freq[ch] = freq.get(ch, 0) + 1
        for i, ch in enumerate(s2):
            temp[ch] = temp.get(ch, 0) + 1
            if temp == freq:
                return True
            if i >= n - 1:
                head = s2[i - n + 1]
                if temp[head] == 1:
                    temp.pop(head)
                else:
                    temp[head] -= 1
        return False

    # https://leetcode.com/problems/number-of-students-doing-homework-at-a-given-time/
    def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
        ans = 0
        for i in range(len(startTime)):
            if queryTime >= startTime[i] and queryTime <= endTime[i]:
                ans += 1
        return ans
    
    # https://leetcode.com/problems/rearrange-words-in-a-sentence/
    def arrangeWords(self, text: str) -> str:
        words = text.lower().split(' ')
        return ' '.join(sorted(words, key=len)).capitalize()

    # https://leetcode.com/problems/people-whose-list-of-favorite-companies-is-not-a-subset-of-another-list/
    def peopleIndexes(self, favoriteCompanies: List[List[str]]) -> List[int]:
        index, n, compSets = {}, 0, []
        for i, p in enumerate(favoriteCompanies):
            compSets.append((i, set()))
            for c in p:
                if c not in index:
                    index[c], n = n, n + 1
                compSets[i][1].add(index[c])
        compSets.sort(key = lambda x : len(x[1]), reverse=True)
        ans = []
        for i in range(len(compSets)):
            found = False
            for j in range(i):
                if len(compSets[j][1]) == len(compSets[i][1]):
                    break
                if compSets[i][1].issubset(compSets[j][1]):
                    found = True
                    break
            if not found:
                ans.append(compSets[i][0])
        return sorted(ans)

    # https://leetcode.com/problems/maximum-number-of-darts-inside-of-a-circular-dartboard/
    def numPoints(self, points: List[List[int]], r: int) -> int:
        ans, n = 1, len(points)
        for (x1, y1), (x2, y2) in itertools.combinations(points, 2):
            d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if d > r * 2: continue
            q = (r * r - d * d / 4) ** 0.5
            x0 = (x1 + x2) / 2.0 + (y1 - y2) / d * q
            y0 = (y1 + y2) / 2.0 - (x1 - x2) / d * q
            ans = max(ans, sum((x0 - x) ** 2 + (y0 - y) ** 2 <= r * r + 0.00001 for x, y in points))
        return ans

    # https://leetcode.com/problems/online-stock-span
class StockSpanner:
    def __init__(self):
        self.stack = []
        self.day = -1
    def next(self, price: int) -> int:
        self.day += 1
        while self.stack and price >= self.stack[-1][1]:
            self.stack.pop()
        ans = self.day - self.stack[-1][0] if self.stack else self.day + 1
        self.stack.append((self.day, price))
        return ans

    # https://leetcode.com/problems/kth-smallest-element-in-a-bst
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        self.k = k
        def helper(root):
            if not root: return None
            res = helper(root.left)
            if res: return res
            if self.k == 1: return root
            self.k -= 1
            res = helper(root.right)
            if res: return res
            return None
        return helper(root).val

    # https://leetcode.com/problems/count-square-submatrices-with-all-ones
    def countSquares(self, matrix: List[List[int]]) -> int:
        ans = sum(matrix[0]) + sum(matrix[i][0] for i in range(1, len(matrix)))
        for i in range(1, len(matrix)):
            for j in range(1, len(matrix[0])):
                if matrix[i][j]:
                    matrix[i][j] = min(matrix[i - 1][j], matrix [i][j - 1], matrix[i - 1][j - 1]) + 1
                ans += matrix[i][j]
        return ans

    # https://leetcode.com/problems/sort-characters-by-frequency
    def frequencySort(self, s: str) -> str:
        freq, ans = {}, ''
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
        for ch in sorted(freq, key=freq.get, reverse=True):
            ans += ch * freq.get(ch)
        return ans    

    # https://leetcode.com/problems/interval-list-intersections/
    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        if not A or not B: return []
        i, j, m, n, ans = 0, 0, len(A), len(B), []
        while i < m and j < n:
            l, r = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
            if l <= r: ans.append([l, r])
            if A[i][1] < B[j][1]: i += 1
            else: j += 1
        return ans

    # https://leetcode.com/problems/check-if-a-word-occurs-as-a-prefix-of-any-word-in-a-sentence/
    def isPrefixOfWord(self, sentence: str, searchWord: str) -> int:
        tokens = sentence.split(' ')
        print(tokens)
        for i, token in enumerate(tokens):
            if token.startswith(searchWord): return i + 1
        return -1

    # https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal
    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
        if not preorder: return None
        root, i = TreeNode(preorder[0]), 1
        while i < len(preorder) and preorder[i] < root.val:
            i += 1
        root.left = self.bstFromPreorder(preorder[1 : i])
        root.right = self.bstFromPreorder(preorder[i :])
        return root

    # https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/
    def maxVowels(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        ans, cnt = 0, 0
        for i, ch in enumerate(s):
            if ch in vowels:
                cnt += 1
            if i >= k - 1:
                ans = max(ans, cnt)
                if s[i - k + 1] in vowels:
                    cnt -= 1
        return ans

    # https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/
    def pseudoPalindromicPaths (self, root: TreeNode) -> int:
        def helper(root, digits):
            if not root: return 0
            digits[root.val] = digits.get(root.val, 0) + 1
            if not root.left and not root.right:
                odd = 0
                for digit in digits:
                    if digits.get(digit) % 2 == 1:
                        odd += 1
                    if odd == 2:
                        break
                ans = 1 if odd < 2 else 0
            else:
                ans = helper(root.left, digits) + helper(root.right, digits)
            if digits[root.val] == 1:
                digits.pop(root.val)
            else:
                digits[root.val] -= 1
            return ans
        return helper(root, {})

    # https://leetcode.com/problems/max-dot-product-of-two-subsequences/
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        m, n = len(nums1), len(nums2)
        f = [[float('-inf')] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                f[i + 1][j + 1] = max(nums1[i] * nums2[j] + max(f[i][j], 0), f[i][j + 1], f[i + 1][j])
        return f[m][n]

    # https://leetcode.com/problems/uncrossed-lines/
    def maxUncrossedLines(self, A: List[int], B: List[int]) -> int:
        m, n = len(A), len(B)
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                f[i + 1][j + 1] = max(f[i][j] + 1 if A[i] == B[j] else 0, f[i + 1][j], f[i][j + 1])
        return f[m][n]

    # https://leetcode.com/problems/contiguous-array
    def findMaxLength(self, nums: List[int]) -> int:
        index, delta, ans = {0: -1}, 0, 0
        for i, num in enumerate(nums):
            delta += 1 if num == 0 else -1
            if delta not in index: index[delta] = i
            else: ans = max(ans, i - index[delta])
        return ans

    # https://leetcode.com/problems/possible-bipartition/
    def possibleBipartition(self, N: int, dislikes: List[List[int]]) -> bool:
        label, g = [0] * (N + 1), {}
        for u, v in dislikes:
            g[u], g[v] = g.get(u, []) + [v], g.get(v, []) + [u]
        def helper(u, color):
            label[u] = color
            for v in g.get(u, []):
                if label[v] == -color: continue
                if label[v] == color or not helper(v, -color): return False
            return True
        for u in range(N):
            if label[u] == 0 and not helper(u, 1): return False
        return True

    # https://leetcode.com/problems/counting-bits
    def countBits(self, num: int) -> List[int]:
        if not num: return [0]
        f, offset = [0, 1], 2
        for i in range(2, num + 1):
            f.append(f[i - offset] + 1)
            if offset * 2 == i + 1: offset *= 2
        return f

    # https://leetcode.com/problems/course-schedule
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g, indegree, ans = {}, defaultdict(lambda: 0), []
        for u, v in prerequisites:
            g[u] = g.get(u, []) + [v]
            indegree[v] += 1
        q = deque()
        for u in range(numCourses):
            if not indegree[u]: q.append(u)
        while q:
            u = q.pop()
            ans += [u]
            for v in g.get(u, []):
                indegree[v] -= 1
                if not indegree[v]: q.append(v)
        return len(ans) == numCourses

    # https://leetcode.com/problems/k-closest-points-to-origin/
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        return heapq.nsmallest(K, points, lambda x: x[0] ** 2 + x[1] ** 2)

    # https://leetcode.com/problems/edit-distance
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        f = [[x for x in range(n + 1)]] + [[x] + [0] * n for x in range(1, m + 1)]
        for i in range(m):
            for j in range(n):
                if word1[i] == word2[j]:
                    f[i + 1][j + 1] = min(f[i][j], min(f[i][j + 1], f[i + 1][j]) + 1)
                else:
                    f[i + 1][j + 1] = min(f[i][j], f[i][j + 1], f[i + 1][j]) + 1
        return f[m][n]

    # https://leetcode.com/problems/invert-binary-tree/
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root: return root
        temp = self.invertTree(root.left)
        root.left = self.invertTree(root.right)
        root.right = temp
        return root

    # https://leetcode.com/submissions/detail/348210543/
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        prev = ListNode(-1)
        prev.next = node
        while node.next:
            node.val, prev, node = node.next.val, node, node.next
        prev.next = None

    # https://leetcode.com/problems/two-city-scheduling/
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        n = len(costs) // 2
        f = [[float('inf')] * (n + 1) for _ in range(2 * n + 1)]
        f[0][0] = 0
        for i in range(2 * n):
            f[i + 1][0] = f[i][0] + costs[i][1]
        for i in range(2 * n):
            for j in range(min(i + 1, n)):
                f[i + 1][j + 1] = min(f[i][j] + costs[i][0], f[i][j + 1] + costs[i][1])
        return f[2 * n][n]

    # https://leetcode.com/problems/reverse-string
    def reverseString(self, s: List[str]) -> None:
        n = len(s)
        for i in range(n // 2):
            ch, s[i] = s[i], s[n - 1 - i]
            s[n - 1 - i] = ch
    
    # https://leetcode.com/problems/random-pick-with-weight/
class Solution:

    def __init__(self, w: List[int]):
        for i in range(1, len(w)): w[i] += w[i - 1]
        self.w = w

    def pickIndex(self) -> int:
        target = random.randint(1, self.w[-1])
        return bisect.bisect_left(self.w, target)

    # https://leetcode.com/problems/queue-reconstruction-by-height/
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people, ans = sorted(people, key = lambda x : (-x[0], x[1])), []
        for p in people:
            ans.insert(p[1], p)
        return ans

    # https://leetcode.com/problems/coin-change-2/
    def change(self, amount: int, coins: List[int]) -> int:
        f = [1] + [0] * amount
        for coin in coins:
            for v in range(coin, amount + 1):
                f[v] += f[v - coin]
        return f[amount]

    # https://leetcode.com/problems/make-two-arrays-equal-by-reversing-sub-arrays/
    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        return sorted(target) == sorted(arr)    

    # https://leetcode.com/problems/check-if-a-string-contains-all-binary-codes-of-size-k/
    def hasAllCodes(self, s: str, k: int) -> bool:
        return len({s[i : i + k] for i in range(len(s) - k + 1)}) == 2 ** k

    # https://leetcode.com/problems/power-of-two/
    def isPowerOfTwo(self, n: int) -> bool:
        return n and n & (-n) == n

    # https://leetcode.com/problems/is-subsequence
    def isSubsequence(self, s: str, t: str) -> bool:
        m, i = len(s), 0
        for ch in t:
            if i == m: break
            if ch == s[i]: i += 1
        return i == m

    # https://leetcode.com/problems/search-insert-position/
    def searchInsert(self, nums: List[int], target: int) -> int:
        return bisect.bisect_left(nums, target)

    # https://leetcode.com/problems/sort-colors
    def sortColors(self, nums: List[int]) -> None:
        l, r, i = 0, len(nums) - 1, 0
        while i <= r:
            if nums[i] == 0:
                nums[i], nums[l] = nums[l], 0
                l += 1
            elif nums[i] == 2:
                nums[i], nums[r] = nums[r], 2
                r -= 1
                i -= 1
            i += 1

    # https://leetcode.com/problems/insert-delete-getrandom-o1/
class RandomizedSet:
    def __init__(self):
        self.index, self.nums = {}, []

    def insert(self, val: int) -> bool:
        if val in self.index: return False
        self.nums += [val]
        self.index[val] = len(self.nums) - 1
        return True

    def remove(self, val: int) -> bool:
        if val not in self.index: return False
        i = self.index[val]
        self.index[self.nums[-1]] = i
        self.index.pop(val)
        self.nums[i] = self.nums[-1]
        self.nums.pop()
        return True

    def getRandom(self) -> int:
        return self.nums[randrange(0, len(self.nums))]

    # https://leetcode.com/problems/largest-divisible-subset
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        if not nums: return nums
        nums.sort()
        n, pos, res = len(nums), 0, []
        f, prev = [1] * n, [-1] * n
        for i in range(1, n):
            for j in range(i):
                if not nums[i] % nums[j] and f[j] + 1 > f[i]:
                    f[i], prev[i] = f[j] + 1, j
            if f[i] > f[pos]: pos = i
        while pos >= 0:
            res.append(nums[pos])
            pos = prev[pos]
        return res

    # https://leetcode.com/problems/cheapest-flights-within-k-stops/
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        adjList = {}
        for u, v, w in flights:
            adjList[u] = adjList.get(u, []) + [(v, w)]
        q, dis = deque([src]), [float("inf")] * n
        dis[src], K = 0, K + 1
        while q and K:
            temp = deepcopy(dis)
            for i in range(len(q)):
                u = q.popleft()
                for v, w in adjList.get(u, []):
                    if dis[u] + w < temp[v]:
                        temp[v] = dis[u] + w
                        q.append(v)
            dis = temp
            K -= 1
        return dis[dst] if dis[dst] != float("inf") else -1
   
    # https://leetcode.com/problems/search-in-a-binary-search-tree
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root: return None
        if root.val == val: return root
        if root.val > val: return self.searchBST(root.left, val)
        return self.searchBST(root.right, val)

    # https://leetcode.com/problems/surrounded-regions
    def solve(self, board: List[List[str]]) -> None:
        if not board: return board
        m, n = len(board), len(board[0])
        def helper(i, j):
            if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != 'O': return
            board[i][j] = 'Y'
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                helper(i + dx, j + dy)
            
        for i in range(n):
            helper(0, i)
            helper(m - 1, i)
        
        for i in range(1, m - 1):
            helper(i, 0)
            helper(i, n - 1)
        
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'Y': board[i][j] = 'O'
                elif board[i][j] == 'O': board[i][j] = 'X'    

    # https://leetcode.com/problems/h-index-ii/
    def hIndex(self, citations: List[int]) -> int:
        n = len(citations)
        start, end = 0, n - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if citations[mid] >= n - mid: end = mid
            else: start = mid
        if n and citations[start] >= n - start: return n - start
        if n and citations[end] >= n - end: return n - end
        return 0

    # https://leetcode.com/problems/longest-duplicate-substring/
    def longestDupSubstring(self, S: str) -> str:
        PRIME = (int)(1e9 + 7)
        def rabinKarp(l):
            seen, h, base = collections.defaultdict(list), 0, pow(26, l - 1, PRIME)
            for i in range(len(S)):
                h = (h * 26 + ord(S[i]) - ord('a')) % PRIME
                if i >= l - 1:
                    if h in seen:
                        substr_i = S[i - l + 1: i + 1]
                        for j in seen[h]:
                            substr_j = S[j - l + 1: j + 1]
                            if substr_i == substr_j: return substr_i
                    seen[h] += [i]
                    h = ((h - (ord(S[i - l + 1]) - ord('a')) * base) % PRIME + PRIME) % PRIME
            return None
        start, end, res = 0, len(S), ""
        while start + 1 < end:
            mid = start + (end - start) // 2
            tmp = rabinKarp(mid)
            if tmp:
                res, start = tmp, mid
            else:
                end = mid
        return res

    # https://leetcode.com/problems/permutation-sequence/
    def getPermutation(self, n: int, k: int) -> str:
        digits, res = [x + 1 for x in range(n)], ""
        n, k = n - 1, k - 1
        while k:
            fac = factorial(n)
            idx, k, n = k // fac, k % fac, n - 1
            digit = digits[idx]
            digits.remove(digit)
            res += chr(digit + ord('0'))
        for digit in digits:
            res += chr(digit + ord('0'))
        return res
