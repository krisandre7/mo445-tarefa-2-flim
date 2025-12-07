iftIntMatrix *iftReadBBSizes(const char *sizes_json_path, iftIntArray **out_labels) {
    iftDict *json = iftReadJson(sizes_json_path);

    iftIntArray *labels = iftGetIntArrayFromDict("labels", json);

    iftIntMatrix *size_mat = iftCreateIntMatrix(3, labels->n);

    for (int o = 0; o < labels->n; o++) {
        char key[512];

        sprintf(key, "data:%d:bounding-box-size", labels->val[o]);
        iftIntArray *sizes = iftGetIntArrayFromDict(key, json);
        iftMatrixElem(size_mat, 0, o) = sizes->val[0];
        iftMatrixElem(size_mat, 1, o) = sizes->val[1];
        iftMatrixElem(size_mat, 2, o) = sizes->val[2];
    }
    iftDestroyDict(&json);

    *out_labels = labels;

    return size_mat;
}


void iftWriteBBSizes(  iftIntMatrix *size_mat, iftIntArray *labels, const char *sizes_json_path) {
    iftDict *json = iftCreateDict();

    iftInsertIntoDict("labels", labels, json);

    for (int o = 0; o < labels->n; o++) {
        char key[512];

        sprintf(key, "data:%d:bounding-box-size", labels->val[o]);
        iftIntArray *sizes = iftCreateIntArray(3);
        sizes->val[0] = iftMatrixElem(size_mat, 0, o);
        sizes->val[1] = iftMatrixElem(size_mat, 1, o);
        sizes->val[2] = iftMatrixElem(size_mat, 2, o);
        iftInsertIntoDict(key, sizes, json);
    }
    iftWriteJson(json, sizes_json_path);
    iftDestroyDict(&json);
}