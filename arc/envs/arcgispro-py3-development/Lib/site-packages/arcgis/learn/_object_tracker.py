try:
    from ._data import _raise_fastai_import_error
    from ._tracking.track_processor import TrackProcessor
    import traceback
    HAS_FASTAI = True
except ImportError:
    import_exception = traceback.format_exc()
    HAS_FASTAI = False

class ObjectTracker:
    """
    Creates ObjectTracker Object.

    =====================   ===========================================
    **Argument**            **Description**
    ---------------------   -------------------------------------------
    tracker                 Required. Returned tracker object from
                            from_model API of object tracking models.
    ---------------------   -------------------------------------------
    detector                Optional. Returned detector object from
                            from_model API of object detection models.
    ---------------------   -------------------------------------------
    tracker_options         Optional dictionary. A dictionary with
                            keys as parameter names and values as
                            parameter values.

                            "detection_interval" refers to
                            the interval in frames at which the detector
                            is invoked.

                            "detection_threshold" refers to
                            the lower threshold for selecting the
                            detections.

                            "detect_track_failure" refers to
                            the flag which enables/disables the logic
                            to detect whether the object appearance
                            has changed detection.

                            "recover_track" refers to the flag which
                            enables/disables track recovery post failure.

                            "stab_period" refers to the number of frames
                            after which post processing starts.

                            "detect_fail_interval" refers to the number
                            of frames after which to detect track failure.

                            "min_obj_size" refers to the size in pixels
                            below which tracking is assumed to have
                            failed.

                            "template_history" refers to the number of
                            frames before the current frame at which
                            template image is fetched.

                            "status_history" refers to the
                            number of frames over which status of the
                            track is used to detect track failure.

                            "status_fail_threshold" refers to the
                            threshold for the ratio between number
                            of frames for which object is searched
                            for and the total number of frames which
                            needs to be crossed for track failure
                            detection.

                            "search_period" refers to the
                            number of frames for which object is
                            searched for before declaring object is
                            lost.

                            "knn_distance_ratio" refers to the
                            threshold for ratio of the distances between
                            template descriptor and the two best matched
                            detection descriptor, used for filtering
                            best matches.

                            "recover_conf_threshold" refers
                            to the minimum confidence value over which
                            recovery logic is enabled.
                            
                            "recover_iou_threshold" refers to the minimum
                            overlap between template and detection for
                            successful recovery.
    =====================   ===========================================

    :returns: `ObjectTracker` Object
    """
    def __init__(self, tracker, detector=None, tracker_options={"detection_interval": 5,
                                                                "detection_threshold": 0.3,
                                                                "detect_track_failure": True,
                                                                "recover_track": True,
                                                                "stab_period": 6,
                                                                "detect_fail_interval": 5,
                                                                "min_obj_size": 10,
                                                                "template_history": 25,
                                                                "status_history": 60,
                                                                "status_fail_threshold": 0.6,
                                                                "search_period": 60,
                                                                "knn_distance_ratio": 0.75,
                                                                "recover_conf_threshold": 0.1,
                                                                "recover_iou_threshold": 0.1}):
        if not HAS_FASTAI:
            _raise_fastai_import_error(import_exception=import_exception)

        self.tracker = tracker
        self.detector = detector
        self.tracks = []
        self.detect_interval = tracker_options.get('detection_interval', 10)
        self.detect_threshold = tracker_options.get('detection_threshold', 0.3)
        processor_options = {}
        processor_options["detect_track_failure"] = tracker_options.get('detect_track_failure', True)
        processor_options["recover_track"] = tracker_options.get('recover_track', True)
        processor_options["stab_period"] = tracker_options.get('stab_period', 6)
        processor_options["detect_fail_interval"] = tracker_options.get('detect_fail_interval', 5)
        processor_options["min_obj_size"] = tracker_options.get('min_obj_size', 10)
        processor_options["template_history"] = tracker_options.get('template_history', 25)
        processor_options["status_history"] = tracker_options.get('status_history', 60)
        processor_options["status_fail_threshold"] = tracker_options.get('status_fail_threshold', 0.6)
        processor_options["search_period"] = tracker_options.get('search_period', 60)
        processor_options["knn_distance_ratio"] = tracker_options.get('knn_distance_ratio', 0.75)
        processor_options["recover_conf_threshold"] = tracker_options.get('recover_conf_threshold', 0.1)
        processor_options["recover_iou_threshold"] = tracker_options.get('recover_iou_threshold', 0.1)

        self.processor = TrackProcessor(processor_options)
        self.frames_processed = 0

    def init(self, frame, detections=None, labels=None, reset=True):
        """
        Initializes tracks based on the detections returned by detector/
        manually fed to the function.

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        frame                   Required numpy array. frame is used to
                                initialize the objects to track.
        ---------------------   -------------------------------------------
        detections              Optional list. A list of bounding box to
                                intialize the tracks.
        ---------------------   -------------------------------------------
        labels                  Optional list. A list of labels corresponding
                                to the detections.
        ---------------------   -------------------------------------------
        reset                   Optional flag. Indicates whether to reset
                                the tracker and remove all existing tracks
                                before initialization.
        =====================   ===========================================

        :returns: list of active track objects
        """
        if detections is None:
            if self.detector is not None:
                predictions, labels, scores = self.detector.predict(frame, return_scores=True)
                detections = self._convert_to_dets(predictions, labels, scores)
        else:
            if labels is None or len(labels) != len(detections):
                labels = ['Object'] * len(detections)

        if detections is not None:
            if len(detections) > 0:
                self.tracks = self.tracker.init(frame, detections, labels, reset)
                track_list = self._convert_tracks_to_list(self.tracks)
                self.processor.init(frame, track_list, reset)

        if reset:
            self.frames_processed = 0
        return self._check_status(self.tracks)

    def update(self, frame):
        """
        Tracks the position of the object in the frame/Image

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        frame                   Required numpy array. frame is the current
                                frame to be used to track the objects.
        =====================   ===========================================

        :returns: list of active track objects
        """
        if self.frames_processed % self.detect_interval == 0 and self.detector is not None:
            self.tracks = self.tracker.update(frame)
            predictions, labels, scores = self.detector.predict(frame, return_scores=True)
            dets = self._convert_to_dets(predictions, labels, scores)
            if len(dets) > 0:
                self.tracks = self.tracker.init(frame, dets)
                tracks_list = self._convert_tracks_to_list(self.tracks)
                self.processor.init(frame, tracks_list, False)
        else:
            if len(self.tracks) > 0:
                self.tracks = self.tracker.update(frame)
                tracks_list = self._convert_tracks_to_list(self.tracks)
                tracks_list = self.processor.update(frame, tracks_list)
                self.tracks = self._update_processed_tracks(tracks_list)

        self.frames_processed = self.frames_processed + 1

        return self._check_status(self.tracks)

    def remove(self, tracks_ids):
        """
        Removes the tracks corresponding to track_ids parameter

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        tracks_ids              Required list. List of track ids to be
                                removed.
        =====================   ===========================================

        """
        self.tracks = self.tracker.remove(tracks_ids)
        self.processor.remove(tracks_ids)

        return self.tracks

    def _convert_to_dets(self, predictions, labels, scores):
        """
        Converts the predictions to 1D list which can be used by
        TrackProcessor

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        predictions             Required list. Predictions which satisfy
                                selection criteria are used.
        ---------------------   -------------------------------------------
        labels                  Optional list. A list containing labels for
                                the predictions.
        ---------------------   -------------------------------------------
        scores                  Required list. A list containing scores for
                                the predictions.
        =====================   ===========================================

        :returns: 1D list with predictions
        """
        if len(predictions) == 0:
            return []

        dets = []
        offset = 0
        max_objects = 1000

        for index in range(offset, min(max_objects, len(predictions) - offset)):
            prediction = predictions[index]
            label = labels[index]
            score = scores[index]

            if score > self.detect_threshold:
                dets.append(prediction)

        return dets

    def _convert_tracks_to_list(self, tracks):
        """
        Converts tracks list to 1D list which can be used by
        TrackProcessor

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        tracks                  Required list. List of tracks to be used by
                                TrackProcessor.
        =====================   ===========================================

        :returns: 1D list with values of track members used by
                  TrackProcessor
        """
        tracks_list=[]
        tracks_list.append(len(tracks))
        tracks_list.append(10)

        for track in tracks:
            tracks_list.append(int(track.id))
            tracks_list.append(str(track.label)) #int(track.class_id)
            tracks_list.append(int(track.age))
            tracks_list.append(float(track.score))
            tracks_list.append(int(track.bbox[0]))
            tracks_list.append(int(track.bbox[1]))
            tracks_list.append(int(track.bbox[2]))
            tracks_list.append(int(track.bbox[3]))
            if track.mask is not None:
                tracks_list.append(255 * track.mask)
            else:
                tracks_list.append(track.mask)
            tracks_list.append(int(track.status))
        return tracks_list

    def _update_processed_tracks(self, tracks_list):
        """
        Uses 1D list returned by TrackProcessor to update list of tracks

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        tracks_list             Required list. 1D list with values of track
                                members.
        =====================   ===========================================

        :returns: list of track objects
        """
        if tracks_list is None:
            return []

        num_dets = tracks_list[0]
        len_det = tracks_list[1]

        for i in range(0, num_dets):
            det_index = (i * len_det) + 2
            id = tracks_list[det_index]

            track_index = -1

            for j, track in enumerate(self.tracks):
                if id == track.id:
                    track_index = j
                    break

            if track_index >= 0:
                self.tracks[track_index].status = tracks_list[det_index + 9]

        return self.tracks

    def _check_status(self, tracks):
        """
        Filters the active tracks using the argument tracks

        =====================   ===========================================
        **Argument**            **Description**
        ---------------------   -------------------------------------------
        tracks                  Required list. List of tracks to be
                                filtered.
        =====================   ===========================================

        :returns: list of active track objects
        """
        active_tracks = filter(lambda track: track.status == 16, tracks)
        return active_tracks
