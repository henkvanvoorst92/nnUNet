import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetDataLoader3D_channel_sampler(nnUNetDataLoaderBase):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 img_gt_sampling_strategy=(False,False,False)):
        super().__init__(data, batch_size, 1, None,
                         True, False,
                         True, sampling_probabilities)

        self.random_gt_sampling, self.random_img_sampling, self.random_gt_img_sampling = img_gt_sampling_strategy
        #self.random_gt_sampling: samples ground truth from channel
        #self.random_img_sampling: samples image from channel
        #self.random_gt_img_sampling: samples both image and ground truth from channel with same index!

    #changed because shape should be different
    def determine_shapes(self):
        data, seg, properties = self._data.load_case(self.indices[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, 1, *self.patch_size)
        return data_shape, seg_shape

    #write a funciton to first random sample a channel

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)

            #sample index
            if len(data.shape)>3 or len(seg.shape)>3:
                ix_seg = None
                if self.random_gt_img_sampling:
                    if seg.shape[0]!=data.shape[0]:
                        raise ValueError("Segmentation and image channels do not match!", seg.shape, data.shape)
                    ix = np.random.choice(np.arange(seg.shape[0]))
                    ix_seg, ix_img = ix, ix
                    # fetch the right channel
                    seg = seg[ix_seg:ix_seg + 1]
                    data = data[ix_img:ix_img + 1]
                elif self.random_gt_sampling:
                    ix_seg = np.random.choice(np.arange(seg.shape[0]))
                    # fetch the right channel
                    seg = seg[ix_seg:ix_seg + 1]
                elif self.random_img_sampling:
                    ix_img = np.random.choice(np.arange(data.shape[0]))
                    data = data[ix_img:ix_img + 1]

                if ix_seg is not None:
                    #alter the properties --> locations of foreground
                    properties = properties.copy() #don't know if this is required
                    #select the foreground coordinates available for sampling
                    coords = properties['class_locations'][1]
                    select_coords = coords[:,0]==ix_seg
                    properties['class_locations'][1] = coords[select_coords]

            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
