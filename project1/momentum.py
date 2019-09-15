new_w_vel = self.momentum*layer.w_vel + self.lr*layer.grad_w
new_b_vel = self.momentum*layer.b_vel + self.lr*layer.grad_b

layer.weights -= new_w_vel
layer.weights -= new_b_vel

layer.update_velocity(new_w_vel, new_b_vel)
