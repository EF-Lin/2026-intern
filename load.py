from obspy import read, Stream


def load_data(files: list | str) -> Stream:
    data = Stream()
    if type(files) == list:
        for f in files:
            data += read(f)
    else:
        data = read(files)
    return data
