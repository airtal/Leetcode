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
