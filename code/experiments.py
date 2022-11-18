from utils import experiment


def main():
    experiment("Pong-v4", 6, epochs=5000, csv_name='results/Pong.csv', vidio_path='./monitor/Pong')
    experiment("Qbert-v4", 6, epochs=5000, csv_name='results/Qbert.csv', vidio_path='./monitor/Qbert')
    experiment("Enduro-v4", 9, epochs=5000, csv_name='results/Enduro.csv', vidio_path='./monitor/Enduro')
    experiment("BeamRider-v4", 9, epochs=5000, csv_name='results/B_Rider.csv', vidio_path='./monitor/B_Rider')
    experiment("Seaquest-v4", 18, epochs=5000, csv_name='results/Seaquest.csv', vidio_path='./monitor/Seaquest')
    experiment("Breakout-v4", 4, epochs=5000, csv_name='results/Breakout.csv', vidio_path='./monitor/Breakout')
    experiment("SpaceInvaders-v4", 6, epochs=5000, csv_name='results/S_Invaders.csv', vidio_path='./monitor/S_Invaders')

    experiment("Breakout-v4", 4, eps_init=1, eps_grad=0.2, eps_min=0.01,
               csv_name='results/Breakout_dif_init_1.csv', vidio_path='./monitor/Breakout_dif_init_1')
    experiment("Breakout-v4", 4, eps_init=0.7, eps_grad=0.2, eps_min=0.01,
               csv_name='results/Breakout_dif_init_2.csv', vidio_path='./monitor/Breakout_dif_init_2')
    experiment("Breakout-v4", 4, eps_init=0.5, eps_grad=0.2, eps_min=0.01,
               csv_name='results/Breakout_dif_init_3.csv', vidio_path='./monitor/Breakout_dif_init_3')
    experiment("Breakout-v4", 4, eps_init=0.3, eps_grad=0.2, eps_min=0.01,
               csv_name='results/Breakout_dif_init_4.csv', vidio_path='./monitor/Breakout_dif_init_4')

    experiment("Breakout-v4", 4, eps_init=0.7, eps_grad=0.01, eps_min=0.01,
               csv_name='results/Breakout_dif_grad_1.csv', vidio_path='./monitor/Breakout_dif_grad_1')
    experiment("Breakout-v4", 4, eps_init=0.7, eps_grad=0.1, eps_min=0.01,
               csv_name='results/Breakout_dif_grad_2.csv', vidio_path='./monitor/Breakout_dif_grad_2')
    experiment("Breakout-v4", 4, eps_init=0.7, eps_grad=0.5, eps_min=0.01,
               csv_name='results/Breakout_dif_grad_3.csv', vidio_path='./monitor/Breakout_dif_grad_3')
    experiment("Breakout-v4", 4, eps_init=0.7, eps_grad=1.0, eps_min=0.01,
               csv_name='results/Breakout_dif_grad_4.csv', vidio_path='./monitor/Breakout_dif_grad_4')


if __name__ == '__main__':
    main()
