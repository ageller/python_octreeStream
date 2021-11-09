from multiprocessing import Process, Manager

def f(msg):
    for ele in msg:
        if 1 in ele:
            ele[1] += "hi"
        if 100 in ele:
            ele[100] += "5"

if __name__ == '__main__':
    manager = Manager()

    msg = manager.list()
    msg.append(manager.dict({ 1: "A", 2: "B"}))
    msg.append(manager.dict({20 : "AAA", 30 : "BBB"}))
    msg.append(manager.dict({100 : "777", 200 : "888"}))

    p1 = Process(target=f, args=(msg,))
    p2 = Process(target=f, args=(msg,))
    p3 = Process(target=f, args=(msg,))

    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()

    for i,d in enumerate(msg):
        print(msg[i])