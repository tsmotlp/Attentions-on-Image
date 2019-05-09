from torch import nn
import torch
import torch.nn.functional as F



device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


class AAConv2d(nn.Module):
    """attention augmented convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dk=40, dv=4, Nh=4, relative=True):
        super(AAConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.relative = relative
        self.general_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels - self.dv,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride, padding=self.padding)
        self.out_conv = nn.Conv2d(in_channels=self.dv, out_channels=self.dv, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding)
        self.qkv_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=2 * self.dk + self.dv, kernel_size=1)

    def forward(self, x):
        # x has shape [B, fin, H, W]
        batch, channels, height, width = x.size()
        conv_out = self.general_conv(x)
        # flat_q, flat_k, flat_v has shape [B, Nh, H*W, dkh or dvh]
        # dkh = dk / Nh
        # dvh = dv / Nh
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        # logits has shape [B, Nh, H*W, H*W]
        logits = torch.matmul(flat_q, flat_k.transpose(2, 3))  # flat_k.transpose(2, 3) shape:[B, Nh, dkh, H*W]
        if self.relative:
            h_real_logits, w_real_logits = self.relative_logits(q)
            logits += h_real_logits
            logits += w_real_logits
        weights = F.softmax(logits, dim=-1)
        # attn_out has shape [B, Nh, H*W, dvh]
        attn_out = torch.matmul(weights, flat_v)  # flat_v has shape [B, Nh, H*W, dvh]
        attn_out = torch.reshape(attn_out, [batch, self.Nh, height, width, self.dv // self.Nh])
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.out_conv(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def combine_heads_2d(self, inputs):
        """combine heads(inverse of split_heads_2d)"""
        batch, Nh, height, width, dvh = inputs.size()
        transposed = inputs.transpose(3, 4).transpose(2, 3)  # shape: [B, Nh, dvh, H, W]
        ret_shape = (batch, Nh * dvh, height, width)
        return torch.reshape(transposed, ret_shape)

    def split_heads_2d(self, inputs, Nh):
        """split channels inpto multiple heads"""
        batch, channels, height, width = inputs.size()
        # print('channels:',channels)
        ret_shape = (batch, Nh, channels // Nh, height, width)
        return torch.reshape(inputs, ret_shape)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        """compute flattened queries, keys and values"""
        batch, channels, height, width = x.size()
        qkv = self.qkv_conv(x)
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)  # [B, Nh, dkh, H, W]
        k = self.split_heads_2d(k, Nh)  # [B, Nh, dkh, H, W]
        v = self.split_heads_2d(v, Nh)  # [B, Nh, dvh, H, W]
        # q = q / sqrt(dkh)
        dkh = dk // Nh
        q = q / dkh ** -0.5
        # calculate flat_q, flat_k, flat_v
        flat_q = torch.reshape(q, (batch, Nh, height * width, dkh))
        flat_k = torch.reshape(k, (batch, Nh, height * width, dkh))
        flat_v = torch.reshape(v, (batch, Nh, height * width, dv // Nh))
        return flat_q, flat_k, flat_v, q, k, v

    def relative_logits(self, q):
        """compute relative position logits"""
        batch, Nh, dkh, height, width = q.size()
        q = torch.transpose(q, 2, 3).transpose(3, 4)  # shape: [B, Nh, H, W, dkh]
        # reltive logits in height dimension
        key_rel_h = nn.Parameter(torch.randn((2 * height - 1, dkh), requires_grad=True)).to(device)
        relative_logits_h = self.relative_logits_1d(q, key_rel_h, height, width, Nh, 'h')
        # relative logits in width dimension
        key_rel_w = nn.Parameter(torch.randn((2 * width - 1, dkh), requires_grad=True)).to(device)
        relative_logits_w = self.relative_logits_1d(q.transpose(2, 3), key_rel_w, height, width, Nh, 'w')
        return relative_logits_h, relative_logits_w

    def relative_logits_1d(self, q, key_rel, H, W, Nh, case):
        """compute relative logits along one dimension"""
        rel_logits = torch.einsum('bhxyd, md->bhxym', q, key_rel)  # shape[B, Nh, H, W, 2W-1]
        # collapse height and heads
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)  # shape: [B, Nh*H, W, W]
        # shape it back and tile height times
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat(1, 1, 1, H, 1, 1)  # shape: [B, Nh, H, H, W, W]
        # reshape for adding to the attension logits
        if case == 'w':  # width dimension: shape [B, Nh, H, H, W, W]
            rel_logits = rel_logits
        elif case == 'h':  # height dimension: shape [B, Nh, W, W, H, H]
            rel_logits = rel_logits.transpose(2, 4).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, inputs):
        """convert tensor from relative to absolute indexing"""
        batch, Nh, L, _ = inputs.size()  # inputs shape [B, Nh, L, 2L-1]
        # pad to shift from relative to absolute indexing
        col_pad = torch.zeros((batch, Nh, L, 1)).to(device)
        x = torch.cat((inputs, col_pad), dim=3)  # shape [B, Nh, L, 2L]
        flat_x = torch.reshape(x, (batch, Nh, L * (2 * L)))  # shape [B, Nh, 2*L*L]
        flat_pad = torch.zeros((batch, Nh, L - 1)).to(device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)  # shape [B, Nh, 2*L*L+L-1]
        # reshape and slice out the padded elements
        final_x = torch.reshape(flat_x_padded, (batch, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


# # Example Code
# import time
#
# tmp = torch.randn((16, 3, 32, 32)).to(device)
# start_time = time.time()
# augmented_conv = AAConv2d(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, Nh=1, relative=True, stride=1,
#                           padding=0).to(device)
# conv_out = augmented_conv(tmp)
# print(conv_out.shape)
#
# print("\nif relative is True, ", time.time() - start_time)