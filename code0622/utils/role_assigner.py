import random
from typing import List
from werewolf_env import Role

class RoleAssigner:
    """角色分配器，用于随机分配游戏角色"""
    
    @staticmethod
    def assign_roles(num_wolves: int, num_villagers: int) -> List[Role]:
        """
        随机分配角色
        
        Args:
            num_wolves: 狼人数量
            num_villagers: 村民数量（不包括预言家）
            
        Returns:
            List[Role]: 随机排列的角色列表
        """
        # 创建角色列表
        roles = [Role.WOLF] * num_wolves + [Role.VILLAGER] * num_villagers + [Role.SEER]
        
        # 随机打乱角色顺序
        # random.shuffle(roles)
        
        return roles 