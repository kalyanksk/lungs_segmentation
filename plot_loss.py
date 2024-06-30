import pandas as pd
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


def plot_iou_score(train_logs_df,valid_logs_df):
    plt.figure(figsize=(20,8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label = 'Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label = 'Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('IoU Score', fontsize=20)
    plt.title('IoU Score Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('assets/iou_score_plot.png')
    plt.show()

def plot_dice_loss(train_logs_df,valid_logs_df):
    plt.figure(figsize=(20,8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label = 'Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label = 'Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Dice Loss', fontsize=20)
    plt.title('Dice Loss Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('assets/dice_loss_plot.png')
    plt.show()


train_logs_df = pd.read_csv('assets/train_log.csv')
valid_logs_df = pd.read_csv('assets/valid_log.csv')

plot_iou_score(train_logs_df,valid_logs_df)
plot_dice_loss(train_logs_df,valid_logs_df)