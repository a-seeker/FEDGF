import xlwt
import time
def write_to_excel(loss_test, acc_test, loss_train, args):
    wb = xlwt.Workbook()
    sh = wb.add_sheet('record')
    sh.write(0, 0, 'test loss')
    sh.write(0, 1, 'test accuracy')
    sh.write(0, 2, 'train loss')
    for row in range(len(loss_test)):
        sh.write(row + 1, 0, loss_test[row])
        sh.write(row + 1, 1, acc_test[row])
        sh.write(row + 1, 2, loss_train[row])
    file = ""
    t = time.localtime()
    if args.noniid_type == 'fedavg':
        file = "fedae_fedavg"
    else:
        file = "fedae_dirichlet{}".format(args.alpha)
    file = "{}_{}_{}_{}_{}_{}_{}.{}".format(file, args.dataset, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min,t.tm_sec, "xls")
    wb.save(file)
    print("file saved in "+file)

if __name__ == '__main__':
    loss_test = [i for i in range(100)]
    acc_test = [i for i in range(100)]
    loss_train = [i for i in range(100)]
    write_to_excel(loss_test, acc_test, loss_test)
