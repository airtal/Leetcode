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
