import can
import time

# bus = can.interface.Bus(channel='can0', bustype='socketcan', bitrate=1000000)
bus = can.interface.Bus(
    bustype='slcan',                    # 串口型
    channel='/dev/cu.usbmodem00000000050C1',   # 你的设备
    bitrate=1000000,                      # 波特率
    # receive_own_messages=True            # 回环测试
)
listener = can.BufferedReader()
notifier = can.Notifier(bus, [listener])

def send_can_message_data(channel, message_id, data):
    
    print(data)
    msg = can.Message(arbitration_id=message_id, data=data, is_extended_id=False)
    try:
        bus.send(msg)
        print(f"Message sent on {channel}: ID={message_id}, Data={data}")
    except can.CanError:
        print("Message NOT sent")

def send_can_message_remote(channel, message_id):
    msg = can.Message(arbitration_id=message_id, is_remote_frame=True, dlc=8)
    try:
        bus.send(msg)
        print(f"Remote frame sent on {channel}: ID={message_id}")
    except can.CanError:
        print("Remote frame NOT sent")

if __name__ == "__main__":
    # Example usage: send message with ID 0x123 and 8 bytes of data on 'can0'
    
    # time.sleep(1)
    pos = [3131, 156, 0, 0]
    delta = 200
    while True:
        try:
            time.sleep(1)
            send_can_message_remote('can0', 0x006)
            received = listener.get_message(timeout=1.0)
            if received:
                print(f"Received message: ID={received.arbitration_id}, Data={list(received.data)}")
                servo_data = list(received.data)
                for i in range(4):
                    print(f'Servo {i+1} Position: {servo_data[2*i]*256 + servo_data[2*i+1]}')
            else:
                print("No message received within timeout period.")
            
            tx_data = []
            for i in range(len(pos)):
                temp = pos[i]
                temp = temp + delta
                tx_data.append(int(temp / 256))
                tx_data.append(int(temp % 256))
            print(tx_data)
            send_can_message_data('can0', 0x006, tx_data)

            time.sleep(1)

            tx_data = []
            for i in range(len(pos)):
                temp = pos[i]
                temp = temp
                tx_data.append(int(temp / 256))
                tx_data.append(int(temp % 256))

            send_can_message_data('can0', 0x006, tx_data)

            time.sleep(1)

        except can.CanError as e:
            print(f"CAN error: {e}")


