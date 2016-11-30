import org.opencv.core.Core;


import java.io.File;


public class Main {
	public static void main(String... args) {
		System.out.println(Core.NATIVE_LIBRARY_NAME);
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		if (args.length == 0) {
			System.err.println("Usage Main /path/to/images");
			System.exit(1);
		}

		//File[] files = new File(args[0]).listFiles();
		showFiles(args[0]);
	}


	public static void showFiles(String file) {
		DetectFaces faces = new DetectFaces();
		//for (File file : files) {
		//if (file.isDirectory()) {
		//    System.out.println("Directory: " + file.getName());
		//    showFiles(file.listFiles()); // Calls same method again.
		//} else {
		System.out.println("File: " + file);
		faces.run(file);
		//}
		//}
	}
}
