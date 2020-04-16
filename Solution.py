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
