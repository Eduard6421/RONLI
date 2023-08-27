import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(description='Model selection script')
    parser.add_argument('--model', type=str, required=True, choices=['cbow','gpt', 'bert', 'bert-spurious', 'bert-cartography', 'bert-cart-cl', 'bert-cart-cl++', 'bert-cart-stra-cl++','bert-multilingual-zero-shot', 'bert-multilingual','bert-oversample'], help='The model to use')
    parser.add_argument('--subtype', type=str,choices=['svc','xgb'], help='Number of epochs to train the model')
    parser.add_argument('--subset', type=str, choices=['easy','ambiguous','hard','easyambiguous'], help='The subset for bert-cartography')

    args = parser.parse_args()
    if args.model == 'bert-cartography' and args.subset is None:
        parser.error("--model bert-cartography requires --subset")

    if args.model == "cbow" and args.subtype is None:
        parser.error("--model cbow requires --subtype")

    print(f'Selected model: {args.model}')
    if args.subset:
        print(f'Selected subset: {args.subset}')

    # Dynamically load the selected model's module
    try:
        model_module = importlib.import_module(f"{args.model}.model")
        func = getattr(model_module, 'train_pipeline')
        func(args)
    except ImportError as e:
        print(e)
        print(f"Failed to import module for model: {args.model}")
        return

    # Run the model's training pipeline
    try:
        model_module.train_pipeline(args.subset if args.model == 'bert-cartography' else None)
    except AttributeError:
        print(f"Failed to run training pipeline for model: {args.model}")


if __name__ == "__main__":
    main()