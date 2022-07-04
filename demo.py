import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from ui.Ui_dialogbot import Ui_MainWindow
import torch
import os
import argparse
from GPT2.interact import Inference

def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--log_path', default='interact.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--model_dir', default='models/model_epoch40_50w/', type=str, required=False, help='对话模型文件夹路径')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=3, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()
    
class My_Ui_MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(My_Ui_MainWindow, self).__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.send)
        args = set_args()
        args.cuda = torch.cuda.is_available() and not args.no_cuda
        device = 'cuda' if args.cuda else 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        self.inference = Inference(args.model_dir, device, args.max_history_len, args.max_len, args.repetition_penalty,
                            args.temperature)
        
    def send(self):
        str = self.lineEdit.displayText()
        self.listWidget.addItem("【usr】:" + str)
        text = self.inference.predict(str)
        self.listWidget.addItem("【robot】:" + text)
        self.listWidget.setCurrentRow(self.listWidget.count() - 1)
        self.lineEdit.setText("")
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = My_Ui_MainWindow()
    myWin.show()
    sys.exit(app.exec_())