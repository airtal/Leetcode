class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict = {}
        for i in range(len(nums)):
            if (target - nums[i]) in dict:
                return [dict[target - nums[i]], i]
            dict[nums[i]] = i

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

    def checkValidString(self, s: str) -> bool:
        low, high = 0, 0
        for ch in s:
            low += 1 if ch is '(' else -1
            high += -1 if ch is ')' else 1
            if (high < 0):
                return False
            low = max(low, 0)
        return low is 0

    def removeElement(self, nums: List[int], val: int) -> int:
        len = 0
        for num in nums:
            if num is not val:
                nums[len], len = num, len + 1
        return len

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
