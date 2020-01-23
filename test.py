

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--a", default=1, help="This is the 'a' variable")

args = parser.parse_args()
a = args.a
print(a)