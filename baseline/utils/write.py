import xlrd
import xlwt

def write_to_excel(args,loss_test, acc_test, loss_train):
    wb = xlwt.Workbook()
    sh = wb.add_sheet('record')
    sh.write(0, 0, 'test loss')
    sh.write(0, 1, 'test accuracy')
    sh.write(0, 2, 'train loss')
    for row in range(len(loss_test)):
        sh.write(row + 1, 0, loss_test[row])
        sh.write(row + 1, 1, acc_test[row])
        sh.write(row + 1, 2, loss_train[row])
    
    if args.noniid_type == 'dirichlet':
        print('saved in ./log/{}_{}_{}.xls'.format(args.flmethod,args.dataset,args.noniid_type))
        wb.save('./log/{}_{}_{}.xls'.format(args.flmethod,args.dataset,args.noniid_type))
    else:
        print('saved in ./log/{}_{}.xls'.format(args.flmethod,args.dataset))
        wb.save('./log/{}_{}.xls'.format(args.flmethod,args.dataset))

if __name__ == '__main__':
    loss_test = [i for i in range(100)]
    acc_test = [i for i in range(100)]
    loss_train = [i for i in range(100)]
    write_to_excel(loss_test, acc_test, loss_test)
