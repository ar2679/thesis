from threading import Lock

output_lock = Lock()
plot_lock = Lock()


def thread_safe_save_csv(df, file_path, **kwargs):
    with output_lock:
        df.to_csv(file_path, **kwargs)
        

def thread_safe_save_plot(plot, file_path):
    with plot_lock:
        plot.savefig(file_path)
        plot.clf()