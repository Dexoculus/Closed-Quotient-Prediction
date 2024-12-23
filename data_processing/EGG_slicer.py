import os
import wave

def slice_wav(input_file, output_dir, person, wav, slice_length=0.1):
    # If directory does not exist, generate directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with wave.open(input_file, 'rb') as wavfile:
        # Extract Audio parameter
        n_channels, sampwidth, framerate, n_frames, comptype, compname = wavfile.getparams()

        # Calculate length of byte per frame
        frame_length = int(framerate * slice_length)

        # Read and Slice files
        for i in range(0, n_frames, frame_length):
            wavfile.setpos(i)
            frames = wavfile.readframes(frame_length)


            if len(frames) >= (frame_length * sampwidth * n_channels):
                # Save Sliced files
                filename = f'{person}_{wav:0>2}_{int(i//frame_length):0>2}.wav'
                output_filename = os.path.join(output_dir, filename)
                with wave.open(output_filename, 'wb') as output:
                    output.setparams((n_channels, sampwidth, framerate, len(frames) // (sampwidth * n_channels), comptype, compname))
                    output.writeframes(frames)

def main():
    person_list = os.listdir(r'C:\Personal_Folder\Vocal_CQ\EGG_data_sep')

    for i, person in enumerate(person_list):
        wavlist = os.listdir(f'C:\\Personal_Folder\\Vocal_CQ\\EGG_data_sep\\{person}')
        output_dir = f'C:\\Personal_Folder\\Vocal_CQ\\EGG_Slice\\{person}'
        # output_dir = f'C:\\Personal_Folder\\Vocal_CQ\\EGG_data'
        for j, wav in enumerate(wavlist):
            path = f'C:\\Personal_Folder\\Vocal_CQ\\EGG_data_sep\\{person}\\{wav}'
            slice_wav(path, output_dir, person, j)

if __name__ == '__main__':
    main()