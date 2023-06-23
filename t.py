from typing import List
def containsDuplicate(nums: List[int]) -> bool:
    return len(set(nums)) != len(nums)
def romanToInt(s: str) -> int:
    roman_numerals = {"M":1000,"CM":900,"D":500,"CD":400,"C":100,"XC":90,"L":50,"XL":40,"X":10,"V":5,"IV":4,"I":1}
    res=0
    i=0
    while i<len(s)-1:
        if i>=len(s)-1:
            break
        if s[i]=='I' and s[i+1]=='V':
            res+=4
            i+=2
            continue
        elif s[i]=='I' and s[i+1]=='X':
            res+=9
            i+=2
            continue
        elif s[i]=='X' and s[i+1]=='L':
            res+=40
            i+=2
            continue
        elif s[i]=='X' and s[i+1]=='C':
            res+=90
            i+=2
            continue
        elif s[i]=='C' and s[i+1]=='D':
            res+=400
            i+=2
            continue
        elif s[i]=='C' and s[i+1]=='M':
            res+=900
            i+=2
            continue
        elif s[i] in roman_numerals.keys():
            res+= roman_numerals[s[i]]
            i += 1
    if i==len(s)-1:
        res+=roman_numerals[s[i]]
    return res


import numpy as np


class Solution:
    def arraySign(self, nums: List[int]) -> int:
        prod=1
        for i in range(len(nums)):
            if nums[i]==0:
                return 0
            if nums[i]<0:
                prod*=-1
            elif nums[i]>0:
                prod*=1
        if prod > 0:
            return 1
        elif prod < 0:
            return -1
s=Solution()

print(s.arraySign([9,72,34,29,-49,-22,-77,-17,-66,-75,-44,-30,-24]))
# print(romanToInt("MCMXCIV"))

