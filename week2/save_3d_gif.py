from matplotlib.animation import FuncAnimation
from week2.src.show_3d_matplotlib import fig, ax, X_sub, Y_sub, Z_sub, color_sub

sc = ax.scatter(X_sub, Y_sub, Z_sub, c=color_sub, s=2)

# 업데이트 함수: 시점을 회전
def update(frame):
    ax.view_init(elev=30, azim=frame)
    return sc, 

# 애니메이션 생성
ani = FuncAnimation(fig, update, frames=360, interval=50, blit=True)

# GIF로 저장
ani.save("C:\\Users\\User\\Projects\\cv-bootcamp\\week2\\pointcloud.gif", writer="pillow")