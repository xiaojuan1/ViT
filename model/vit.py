import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224,patch_size=16,in_c=3,embed_dim=768,norm_layer=None):
        #img大小 patchsize 每个patch的大小
        super().__init__()
        img_size=(img_size,img_size)#将输入的图像大小变为2维元组
        patch_size=(patch_size,patch_size)
        self.img_size=img_size
        self.patch_size=patch_size
        self.gird_size=(img_size[0]//patch_size[0],img_size[1]//patch_size[1])#patch的网格大小 14*14
        self.num_patchs=self.gird_size[0]*self.gird_size[1]#14*14=196 patch总数

        self.proj=nn.Conv2d(in_c,embed_dim,kernel_size=patch_size,stride=patch_size)#B,3,224,224->B,768,14,14
        self.norm=norm_layer(embed_dim)if norm_layer else nn.Identity()#里面每一个 patch token 的最后一维 768 做 norm
    def forward(self,x):
        B,C,H,W=x.shape#获取我们输入张量的形状
        assert H==self.img_size[0] and W==self.img_size[1],\
        f"输入图像的大小{H}*{W}和模型期望大小{self.img_size[0]}*{self.img_size[1]}不匹配"
        #B,3,224,224->B,768,14,14对2和3展平 -> B,768,196->B,196,768
        x=self.proj(x).flatten(2).transpose(1,2)
        x=self.norm(x)#若有归一化层,则使用
        return x
def drop_path(x, drop_prob: float = 0., training: bool = False):#对同一个batch里面的每个样本,以drop_prob的概率随机丢弃路径,在训练时使用,测试时不使用
    """
    Drop paths（随机深度）每个样本（在残差块的主路径中应用时）。
    这个实现类似于 DropConnect，用于 EfficientNet 等网络，但名字不同，DropConnect 是另一种形式的 dropout。
    链接中有详细的讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    我们使用 'drop path' 而不是 'DropConnect' 来避免混淆，并将参数名用 'survival rate' 来代替。
    
    参数：
    - x: 输入张量。
    - drop_prob: 丢弃路径的概率。
    - training: 是否处于训练模式。

    返回：
    - 如果不在训练模式或丢弃概率为 0，返回输入张量 x；
    - 否则，返回经过丢弃操作后的张量。
    """
    if drop_prob == 0. or not training:  # 如果丢弃概率为 0 或不处于训练模式，直接返回原始输入
        return x
    keep_prob = 1 - drop_prob  # 保持路径的概率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 生成与 x 的维度匹配的形状，只保持 batch 维度
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 生成一个与 x 大小相同的随机张量
    random_tensor.floor_()  # 将随机张量二值化（小于 keep_prob 的值为 0，其他为 1）
    output = x.div(keep_prob) * random_tensor  # 将输入 x 缩放并与随机张量相乘，实现部分路径的丢弃
    return output  # 返回经过 drop path 操作后的张量

class DropPath(nn.Module):
    """
    Drop paths（随机深度）每个样本（在残差块的主路径中应用时）。
    
    这是一个 PyTorch 模块，用于在训练期间随机丢弃某些路径，以增强模型的泛化能力。
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()  # 调用父类 nn.Module 的构造函数
        self.drop_prob = drop_prob  # 初始化丢弃概率

    def forward(self, x):
        """
        前向传播函数，调用 drop_path 函数。
        
        参数：
        - x: 输入张量。

        返回：
        - 经过 drop path 操作后的张量。
        """
        return drop_path(x, self.drop_prob, self.training)  # 调用上面定义的 drop_path 函数
    
class Attention(nn.Module):
    def __init__(self,
                 dim,#输入的token维度768,
                 num_heads=8,
                 qkv_bias=False,#生成qkv的时候是否添加bias
                 qk_scale=None,#用于缩放QK的系数,如果None,则使用1/sqrt(embed_dim_pre_head)
                 atte_drop_ration=0.,#注意力分数的dropout的比率,防止过拟合
                 proj_drop_ration=0.):#最终投影层的dropout比例)
        super().__init__()
        self.num_heads=num_heads#多头注意力的头数
        head_dim=dim//num_heads#每个头的维度
        self.scale=qk_scale or head_dim**-0.5#缩放系数,如果没有指定,则使用1/sqrt(head_dim)
        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias)#生成qkv的线性层,输入维度是dim,输出维度是dim*3,全连接层生成QKV,为了并行计算,参数更少
        self.attn_drop=nn.Dropout(atte_drop_ration)#注意力分数的dropout层
        #将每个head得到的输出进行concat拼接后,再通过一个线性层进行投影,得到最终的输出
        self.proj=nn.Linear(dim,dim)#最终投影层,输入输出维度都是dim
        self.proj_drop=nn.Dropout(proj_drop_ration)#最终投影层

    def forward(self,x):
        B,N,C=x.shape#Batch,num_patchs+1,emd_dim 这个1是我们的class token
        #B N 3*C ->B N 3,num_heads,head_dim ->3 B num_heads N head_dim 3是为了后续拆分qkv方便
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        #取qkv,形状是B num_heads N head_dim
        q,k,v=qkv[0],qkv[1],qkv[2]#分别得到q,k,v
        #计算qv点积注意力 最后两个维度做一个调换
        attn=(q @ k.transpose(-2,-1)) * self.scale#B num_heads N N
        attn=attn.softmax(dim=-1)#对最后一个维度做softmax,得到注意力权重
        #注意力权重对v进行加权求和
        # attn @ v：B, num_heads, N, head_dim
        # transpose：B,N，self.num_heads,C//self.num_heads
        # reshape：B,N,C,将最后两个维度信息拼接，合并多个头输出，回到总的嵌入维度
        x=(attn @ v).transpose(1,2).reshape(B,N,C)#B num_heads N head_dim ->B N num_heads head_dim ->B N C
        x=self.proj(x)#通过最终投影层,之前reshape是拼接起来的,现在要融合到一起
        x = self.proj_drop(x) # 防止过拟合
        return x
    
class MLP(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        # in_features输入的维度  hidden_features 隐藏层的维度 通常为in_features的4倍，out_features维度 通常与in_features相等
        super().__init__()
        out_features=out_features or in_features#如果没有指定输出维度,则使用输入维度
        hidden_features=hidden_features or in_features#如果没有指定隐藏层维度,则使用输入维度
        self.fc1=nn.Linear(in_features,hidden_features)#第一层全连接层,输入维度是in_features,输出维度是hidden_features
        self.act=act_layer()#激活函数
        self.fc2=nn.Linear(hidden_features,out_features)#第二层全连接层,输入维度是hidden_features,输出维度是out_features
        self.drop=nn.Dropout(drop)#dropout层
    def forward(self,x):
        x=self.fc1(x)#通过第一层全连接层
        x=self.act(x)#通过激活函数
        x=self.drop(x)#通过dropout层
        x=self.fc2(x)#通过第二层全连接层
        x=self.drop(x)#通过dropout层
        return x 
    
class Block(nn.Module):
    def __init__(self,
                 dim,#每个token的维度 768
                 num_heads,#多头注意力的头数
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,#dropout比率,在多头注意力的linear后使用的dropout
                 attn_drop=0.,#注意力分数的dropout比率
                 drop_path=0.,#stochastic depth的drop路径比率,在训练时随机丢弃一些块,以防止过拟合
                 act_layer=nn.GELU,#激活函数
                 norm_layer=nn.LayerNorm):#正则化层
        super(Block,self).__init__()
        self.norm1=norm_layer(dim)#第一个正则化层,在多头注意力之前使用
        #实例化多头注意力机制
        self.attn=Attention(dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
                            attn_drop=attn_drop,proj_drop_ration=drop_ratio)#多头注意力层
        
        self.drop_path=DropPath(drop_ratio) if drop_path>0. else nn.Identity()#如果drop_path大于0,则使用DropPath,否则使用Identity不做任何更改
        self.norm2=norm_layer(dim)#第二个正则化层,在MLP之前使用 
        mlp_hidden_dim=int(dim*mlp_ratio)#MLP隐藏层的维度,通常是输入维度的4倍
        #定义mlp层,传入dim=mlp_hidden_dim
        self.mlp=MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_ratio)#MLP层
    def forward(self,x):
        x=x+self.drop_path(self.attn(self.norm1(x)))#多头注意力层的输出加上输入,形成残差连接,并通过drop_path进行随机丢弃
        x=x+self.drop_path(self.mlp(self.norm2(x)))#MLP层的输出加上输入,形成残差连接,并通过drop_path进行随机丢弃
        return x
class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_c=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,#transformer encoderd的层数
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,#如果不为None,则在分类头之前添加一个线性层,将嵌入维度转换为representation_size
                 distilled=False,#是否使用蒸馏token,如果为True,则在输入中添加一个额外的token用于蒸馏训练???不太懂蒸馏训练
                 drop_ratio=0.,
                 attn_drop=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=nn.LayerNorm,
                 act_layer=None):
        super(VisionTransformer,self).__init__()
        self.num_classes=num_classes
        self.num_features=self.embed_dim=embed_dim
        self.num_tokens=2 if distilled else 1#如果使用蒸馏num_tokens=2,否则为1
        #设置一个比较小的参数防止除以0
        norm_layer=norm_layer or partial(nn.LayerNorm,eps=1e-6)#如果没有指定norm_layer,则使用LayerNorm,并设置eps参数为1e-6
        act_layer=act_layer or nn.GELU#如果没有指定act_layer,则使用GELU激活函数
        self.patch_embed=embed_layer(img_size=img_size,patch_size=patch_size,in_c=in_c,embed_dim=embed_dim,norm_layer=norm_layer)#实例化patch embedding层
        num_patches=self.patch_embed.num_patchs#计算patch的数量14 *14=196
        #使用nn.Parameter构建可训练的参数用于零矩阵初始化目的一个是batch维度,另一个是token维度,最后一个是嵌入维度
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))#分类token,形状为1,1,embed_dim
        self.dist_token=nn.Parameter(torch.zeros(1,1,embed_dim)) if distilled else None#蒸馏token,如果使用蒸馏则形状为1,1,embed_dim,否则为None这里面我们不会用到
        #pos_embed大小与concat拼接后的大小一致,197 768
        self.pos_embed=nn.Parameter(torch.zeros(1,num_patches+self.num_tokens,embed_dim))#位置编码,形状为1(占位符),num_patches+num_tokens,embed_dim
        self.pos_embed_drop=nn.Dropout(drop_ratio)#位置编码的dropout层
        # 根据传入的drop_path_ratio构建等差序列从0到drop_path_ratio，有depth个元素.浅层不丢弃深层才丢弃 
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        #使用nn.Sequential将列表中所有模块打包成一个整体
        self.block=nn.Sequential(*[
            Block(dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,
                  drop_ratio=drop_ratio,attn_drop=attn_drop,drop_path=dpr[i],norm_layer=norm_layer,act_layer=act_layer)
            for i in range(depth)
        ])

        self.norm=norm_layer(embed_dim)#通过transformer的layernormer
        #要不要在最终分类前，加一个“表示层”
        #如果用了 pre_logits，那后面模型真正拿来分类的特征维度就不再是 embed_dim，而是：representation_size
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc",nn.Linear(embed_dim,representation_size)),
                ("act",nn.Tanh())
            ]))
        else:
            self.has_logits=False
            self.pre_logits=nn.Identity() # pre_logits不做任何处理
        #分类头
        self.head=nn.Linear(self.num_features,num_classes) if num_classes>0 else nn.Identity()#分类头,如果num_classes大于0,则使用线性层,否则使用Identity不做任何处理
        # self.head_dist = None
        # if distilled:
        #     self.head_dist = nn.Linear(self.embed_dim,self.num_classes) if num_classes>0 else nn.Identity()

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed,std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token,std=0.02)
        
        nn.init.trunc_normal_(self.cls_token,std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self,x):
        # B C H W -> B num_patches embed_dim
        x = self.patch_embed(x)
        # 1,1,768->B,1,768
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)
        # 如果dist_token存在则拼接dist_token和cls_token,否则只拼接cls_token和输入的patch特征x
        if self.dist_token is None:
            x = torch.cat((cls_token,x),dim=1) # B 197 768 在维度1上面拼接
        else:
            x = torch.cat((cls_token,self.dist_token.expand(x.shape[0],-1,-1),x),dim=1)
        
        x = self.pos_drop(x+self.pos_embed)
        x = self.block(x)
        x = self.norm(x)
        if self.dist_token is None: # dist_token为None 提取cls_token对应的输出
            return self.pre_logits(x[:,0])
        else:
            return x[:,0],x[:,1]#CLS Token 的输出（代表整张图像的特征）Dist Token 的输出（用于蒸馏）
        
    def forward(self,x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            # 分别通过head和head-dist进行预测
            x , x_dist = self.head(x[0]),self.head_dist(x[1])
            # 如果是训练模式且不是脚本模式
            if self.training and not torch.jit.is_scripting():
                # 则返回两个头部的预测结果
                return x ,x_dist
        else:
            x = self.head(x) # 最后的linear 全连接层
        return x 
    
    
def _init_vit_weights(m):
    # 判断模块m是否是nn.linear
    if isinstance(m,nn.Linear):
        nn.init.trunc_normal_(m.weight,std=.01)
        if m.bias is not None: # 如果线性层存在偏置项
            nn.init.zeros_(m.bias)

    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode="fan_out") # 对卷积层的权重做一个初始化 适用于卷积
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m,nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight) # 对层归一化的权重初始化为1

def vit_base_patch16_224(num_classes:int = 1000,pretrained=False):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model

