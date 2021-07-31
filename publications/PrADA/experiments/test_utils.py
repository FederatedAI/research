from utils import test_classifier


def test_model(task_id, init_model, trained_model_root_folder, target_test_loader):
    print("[INFO] load trained model")
    init_model.load_model(root=trained_model_root_folder,
                          task_id=task_id,
                          load_global_classifier=True,
                          timestamp=None)

    init_model.print_parameters()

    print("[INFO] Run test")
    _, auc, ks = test_classifier(init_model, target_test_loader, "test")
    print(f"[INFO] test auc:{auc}, ks:{ks}")
